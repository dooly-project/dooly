import subprocess
import sys
import textwrap
import types
import unittest

import torch

from doolyprof.tracer.hooks import (
    _iter_module_model_config_values,
    _original_tensor_unflatten,
    _wrap_arange,
    _wrap_unflatten,
)
from doolyprof.profiler.finput import ProfileInput
from doolyprof.profiler.module import ModuleInfo
from doolyprof.profiler.parser import Event, OperationHierarchy
from doolyprof.profiler.resolver import (
    ModuleResolver,
    _is_aggregate_model_name,
    _is_attention_replay_target,
    _tensor_call_signature,
)
from doolyprof.profiler.taint import DimensionType
from doolyprof.tracer.propagation import propagate_taints
from doolyprof.tracer.registry import register_taint
from doolyprof.tracer.sglang_subprocess import (
    _retag_forward_batch_runtime_tensors,
    _retag_runtime_tensor,
)
from doolyprof.tracer.tensor import TaintedTensor
from doolyprof.tracer.types import DimTaint, Taint, TaintedInt


def profile_input(shape, dtype="torch.bfloat16"):
    return ProfileInput(
        type="Tensor",
        dtype=dtype,
        dimensions=[DimensionType.MODEL_CONFIG] * len(shape),
        actual_shape=list(shape),
    )


class SGLangFakeTensorParallelTests(unittest.TestCase):
    def test_moe_uses_fake_tensor_parallel_group(self):
        probe = textwrap.dedent(
            """
            from doolyprof.profiler.utils.fake_tp_sglang import apply_fake_tp_patches_sglang

            apply_fake_tp_patches_sglang(4)

            import torch.distributed as torch_dist
            import sglang.srt.distributed as dist
            import sglang.srt.distributed.communication_op as comm_op
            import sglang.srt.distributed.parallel_state as ps
            from sglang.srt.layers.moe.fused_moe_triton import layer as fused_moe_layer

            assert ps.get_tp_group().world_size == 4
            assert ps.get_moe_tp_group().world_size == 4
            assert ps.get_moe_tensor_parallel_world_size() == 4
            assert ps.get_moe_tensor_parallel_rank() == 0
            assert dist.get_moe_tensor_parallel_world_size() == 4
            assert fused_moe_layer.get_moe_tensor_parallel_world_size() == 4
            assert fused_moe_layer.get_moe_tensor_parallel_rank() == 0
            assert fused_moe_layer.get_tp_group().world_size == 4
            assert comm_op.get_moe_tp_group().world_size == 4
            assert torch_dist.monitored_barrier() is None
            """
        )

        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )


class IntImportHookTests(unittest.TestCase):
    def test_repeated_imports_do_not_rescan_unchanged_modules(self):
        probe = textwrap.dedent(
            """
            import sys
            import types

            import doolyprof.tracer.hooks as hooks

            hooks.patch_int_in_packages(["dooly_test_int_patch."])
            original_sweep = hooks._sweep_and_patch_int
            sweep_calls = 0

            def counted_sweep(*args, **kwargs):
                global sweep_calls
                sweep_calls += 1
                return original_sweep(*args, **kwargs)

            hooks._sweep_and_patch_int = counted_sweep
            for _ in range(100):
                __import__("errno")
            assert sweep_calls == 0, sweep_calls

            module_name = "dooly_test_int_patch.lazy_module"
            lazy_module = types.ModuleType(module_name)
            sys.modules[module_name] = lazy_module
            __import__("errno")

            assert sweep_calls == 1, sweep_calls
            assert lazy_module.int is hooks._preserving_int
            """
        )

        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )


class SGLangRuntimeTaintTests(unittest.TestCase):
    def test_hybrid_kv_pool_capacities_are_taint_sources(self):
        probe = textwrap.dedent(
            """
            from types import SimpleNamespace

            from doolyprof.tracer.sglang_subprocess import patch_server_args
            from doolyprof.tracer.types import TaintedInt

            patch_server_args(SimpleNamespace())

            from sglang.srt.model_executor.model_runner import ModelRunner

            runner = object.__new__(ModelRunner)
            runner.full_max_total_num_tokens = 11000
            runner.swa_max_total_num_tokens = 8800

            full_slots = runner.full_max_total_num_tokens + 1
            swa_slots = runner.swa_max_total_num_tokens + 1

            assert isinstance(full_slots, TaintedInt)
            assert isinstance(swa_slots, TaintedInt)
            assert str(full_slots.taint) == "MAX_NUM_TOKS"
            assert str(swa_slots.taint) == "MAX_NUM_TOKS"
            """
        )

        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )

    def test_runtime_boundary_overrides_value_registry_collision(self):
        tensor = TaintedTensor(torch.arange(4), [Taint("MODEL_CONFIG")])

        retagged = _retag_runtime_tensor(tensor, "NUM_TOKS")

        self.assertIs(retagged, tensor)
        self.assertEqual(str(retagged._dim_taints[0].taint), "NUM_TOKS")

    def test_runtime_boundary_wraps_plain_tensor(self):
        tensor = torch.arange(4)

        retagged = _retag_runtime_tensor(tensor, "NUM_TOKS")

        self.assertIsInstance(retagged, TaintedTensor)
        self.assertEqual(str(retagged._dim_taints[0].taint), "NUM_TOKS")

    def test_runtime_boundary_only_retags_workload_axis(self):
        tensor = TaintedTensor(
            torch.zeros(1, 4, 8),
            [Taint("MODEL_CONFIG"), Taint("MODEL_CONFIG"), Taint("MODEL_CONFIG")],
        )

        retagged = _retag_runtime_tensor(tensor, "NUM_TOKS", dim=1)

        self.assertIsNone(retagged._dim_taints[0])
        self.assertEqual(str(retagged._dim_taints[1].taint), "NUM_TOKS")
        self.assertEqual(str(retagged._dim_taints[2].taint), "MODEL_CONFIG")

    def test_forward_batch_copy_is_retagged_after_buffering(self):
        model_config = Taint("MODEL_CONFIG")
        batch = type("Batch", (), {})()
        batch.input_ids = TaintedTensor(torch.arange(4), [model_config])
        batch.positions = TaintedTensor(
            torch.arange(4).reshape(1, 4),
            [model_config, model_config],
        )
        batch.out_cache_loc = TaintedTensor(torch.arange(4), [model_config])
        batch.seq_lens = TaintedTensor(torch.ones(1), [model_config])

        _retag_forward_batch_runtime_tensors(batch)

        self.assertEqual(str(batch.input_ids._dim_taints[0].taint), "NUM_TOKS")
        self.assertIsNone(batch.positions._dim_taints[0])
        self.assertEqual(str(batch.positions._dim_taints[1].taint), "NUM_TOKS")
        self.assertEqual(str(batch.out_cache_loc._dim_taints[0].taint), "NUM_TOKS")
        self.assertEqual(str(batch.seq_lens._dim_taints[0].taint), "NUM_REQS")


class TaintPropagationCollisionTests(unittest.TestCase):
    def test_first_operand_wins_on_same_aligned_axis(self):
        runtime = (Taint("NUM_TOKS"), Taint("MODEL_CONFIG"))
        stale = (Taint("MODEL_CONFIG"), Taint("MODEL_CONFIG"))

        result = propagate_taints(
            [((4, 1152), stale), ((4, 1152), runtime)],
            (4, 1152),
        )

        self.assertEqual(str(result[0].taint), "MODEL_CONFIG")
        self.assertEqual(str(result[1].taint), "MODEL_CONFIG")

        result = propagate_taints(
            [((4, 1152), runtime), ((4, 1152), stale)],
            (4, 1152),
        )

        self.assertEqual(str(result[0].taint), "NUM_TOKS")
        self.assertEqual(str(result[1].taint), "MODEL_CONFIG")

    def test_ambiguous_size_only_fallback_does_not_guess(self):
        result = propagate_taints(
            [
                ((4, 256), (Taint("MODEL_CONFIG"), Taint("MODEL_CONFIG"))),
                ((8, 4), (Taint("NUM_TOKS"), Taint("NUM_TOKS"))),
            ],
            (4,),
        )

        self.assertIsNone(result[0])

    def test_aligned_singleton_is_not_filled_from_another_input(self):
        result = propagate_taints(
            [
                (
                    (1, 4, 4, 256),
                    (
                        None,
                        Taint("MODEL_CONFIG"),
                        Taint("NUM_TOKS"),
                        Taint("MODEL_CONFIG"),
                    ),
                ),
                (
                    (1, 1, 4, 256),
                    (
                        None,
                        Taint("MODEL_CONFIG"),
                        Taint("NUM_TOKS"),
                        Taint("MODEL_CONFIG"),
                    ),
                ),
            ],
            (1, 4, 4, 256),
        )

        self.assertIsNone(result[0])
        self.assertEqual(str(result[1].taint), "MODEL_CONFIG")
        self.assertEqual(str(result[2].taint), "NUM_TOKS")

    def test_nested_dim_taint_is_normalized(self):
        nested = DimTaint.from_taint(DimTaint.from_taint(Taint("NUM_TOKS")))

        result = propagate_taints([((4,), (nested,))], (4,))

        self.assertEqual(str(result[0].taint), "NUM_TOKS")

    def test_arange_uses_registered_source_dimension(self):
        register_taint(258, Taint("MODEL_CONFIG"))

        result = _wrap_arange(torch.arange)(0, 258, 2)

        self.assertEqual(tuple(result.shape), (129,))
        self.assertEqual(str(result._dim_taints[0].taint), "MODEL_CONFIG")

    def test_alibi_head_partitions_are_model_config(self):
        module = types.SimpleNamespace(tp_q_head_num=28, tp_k_head_num=4)

        values = _iter_module_model_config_values(module)

        self.assertIn(28, values)
        self.assertIn(16, values)
        self.assertIn(12, values)

    def test_unflatten_preserves_source_axis_roles_on_equal_sizes(self):
        tensor = TaintedTensor(
            torch.randn(4, 1024),
            [Taint("NUM_TOKS"), Taint("MODEL_CONFIG")],
        )

        result = _wrap_unflatten(_original_tensor_unflatten)(
            tensor,
            -1,
            (TaintedInt(4, Taint("MODEL_CONFIG")), 256),
        )

        self.assertEqual(tuple(result.shape), (4, 4, 256))
        self.assertEqual(str(result._dim_taints[0].taint), "NUM_TOKS")
        self.assertEqual(str(result._dim_taints[1].taint), "MODEL_CONFIG")
        self.assertEqual(str(result._dim_taints[2].taint), "MODEL_CONFIG")


class SameClassNorm(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(width))

    def forward(self, value):
        return value * self.weight


class AttentionParent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_norm = SameClassNorm(256)


class ResolverModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = AttentionParent()
        self.hidden_norm = SameClassNorm(1152)


def hierarchy_with_kernel(parent_name="AttentionParent"):
    parent = Event("module", parent_name, 0, 10, 1, None, None, None)
    anchor = Event("module", "SameClassNorm", 1, 5, 2, None, None, None)
    anchor.parent = parent
    anchor.descendant_kernels = frozenset({"norm_kernel"})
    kernel = Event("kernel", "norm_kernel", 2, 1, 3, None, None, None)
    return OperationHierarchy(anchor, parent, parent, [kernel], [])


class ResolverSelectionTests(unittest.TestCase):
    def make_resolver(self):
        model = ResolverModel()
        vlp = types.SimpleNamespace(model=model)
        resolver = ModuleResolver(vlp=vlp, importer=None)
        resolver._name_to_module["SameClassNorm"] = [
            "hidden_norm",
            "attn.q_norm",
        ]
        return resolver, model

    def module(self, width):
        return ModuleInfo(
            module_name="AttentionParent",
            operation_name="SameClassNorm",
            inputs=[profile_input((4, width))],
            is_module=True,
            hierarchy=hierarchy_with_kernel(),
        )

    def test_model_config_hidden_norm_keeps_kernel_selected_path(self):
        resolver, model = self.make_resolver()
        resolver._pick_path_by_kernels = (
            lambda paths, module, kernels: "hidden_norm"
        )

        module = self.module(1152)
        selected = resolver._import_module(module)

        self.assertIs(selected, model.hidden_norm)
        self.assertEqual(module._resolved_path, "hidden_norm")

    def test_cache_separates_model_config_input_widths(self):
        resolver, model = self.make_resolver()
        picks = []

        def pick(paths, module, kernels):
            picks.append(_tensor_call_signature(module.inputs))
            width = module.inputs[0].actual_shape[-1]
            return "hidden_norm" if width == 1152 else "attn.q_norm"

        resolver._pick_path_by_kernels = pick
        hidden = resolver._import_module(self.module(1152))
        query = resolver._import_module(self.module(256))
        hidden_again = resolver._import_module(self.module(1152))

        self.assertIs(hidden, model.hidden_norm)
        self.assertIs(query, model.attn.q_norm)
        self.assertIs(hidden_again, model.hidden_norm)
        self.assertEqual(len(picks), 2)

    def test_attention_replay_classification_uses_target_operation(self):
        norm = self.module(1152)
        attention = ModuleInfo(
            module_name="Gemma3DecoderLayer",
            operation_name="Gemma3Attention",
            inputs=[profile_input((1, 1152))],
            is_module=True,
        )

        self.assertFalse(_is_attention_replay_target(norm))
        self.assertTrue(_is_attention_replay_target(attention))

    def test_aggregate_model_names_are_not_replay_targets(self):
        self.assertTrue(_is_aggregate_model_name("Gemma3TextModel"))
        self.assertTrue(_is_aggregate_model_name("LlamaForCausalLM"))
        self.assertFalse(_is_aggregate_model_name("Gemma3DecoderLayer"))
        self.assertFalse(_is_aggregate_model_name("RadixAttention"))

if __name__ == "__main__":
    unittest.main()
