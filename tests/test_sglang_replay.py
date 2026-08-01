import types
import unittest

import torch

from dooly.taintracer.hooks import (
    _iter_module_model_config_values,
    _original_tensor_unflatten,
    _wrap_arange,
    _wrap_unflatten,
)
from dooly.profiler.finput import ProfileInput
from dooly.profiler.module import ModuleInfo
from dooly.profiler.parser import Event, OperationHierarchy
from dooly.profiler.resolver import (
    ModuleResolver,
    _is_aggregate_model_name,
    _is_attention_replay_target,
    _tensor_call_signature,
)
from dooly.profiler.taint import DimensionType
from dooly.taintracer.propagation import propagate_taints
from dooly.taintracer.registry import register_taint
from dooly.taintracer.sglang_subprocess import (
    _retag_forward_batch_runtime_tensors,
    _retag_runtime_tensor,
)
from dooly.taintracer.tensor import TaintedTensor
from dooly.taintracer.types import DimTaint, Taint, TaintedInt


def profile_input(shape, dtype="torch.bfloat16"):
    return ProfileInput(
        type="Tensor",
        dtype=dtype,
        dimensions=[DimensionType.MODEL_CONFIG] * len(shape),
        actual_shape=list(shape),
    )


class SGLangRuntimeTaintTests(unittest.TestCase):
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
