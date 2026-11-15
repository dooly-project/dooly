"""
Fake Tensor Parallelism (TP) for single-GPU profiling.

Patches vLLM's parallel_state, distributed communication, and linear layer
modules so that models behave as if running under TP > 1, while actually
executing on a single GPU.

IMPORTANT: This module must be imported BEFORE any vLLM modules so that
all cached references to parallel_state functions see the patched versions.

Usage:
    # In your entry-point script, set the env var before importing anything else:
    os.environ["VLLM_FAKE_TP"] = "2"

    # Then import this module to apply patches:
    from doolyprof.profiler.utils.fake_tp import apply_fake_tp_patches, get_fake_tp_size
"""

import os
import torch
from torch.profiler import record_function
from doolyprof.tracer.tensor import TaintedTensor
from doolyprof.tracer.types import TaintedInt, TaintedShape

from doolyprof.profiler.utils.comm_emulation import format_tensor_shape  # noqa: E402,F401


def get_fake_tp_size() -> int:
    """Return the fake TP size from the VLLM_FAKE_TP environment variable."""
    return int(os.environ.get("VLLM_FAKE_TP", "1"))


def get_fake_ep() -> bool:
    """Whether the fake TP group also serves as the expert-parallel group
    (VLLM_FAKE_EP=1).

    vLLM's MoE layer derives its parallel layout from the TP world size and
    ``ParallelConfig.enable_expert_parallel`` (``FusedMoEParallelConfig.make``):
    with EP on and data parallelism off, ``ep_size = tp_size``, every rank holds
    ``num_experts / ep_size`` whole experts and the expert weights are not
    TP-sharded. Because the TP world size is the patched fake one, passing
    ``enable_expert_parallel=True`` to the engine is all fake EP needs: rank 0
    of a TP+EP deployment is built and run on one GPU. The MoE output is then
    all-reduced over the TP group, which fake TP records as a COMM all_reduce,
    exactly as vLLM does for TP+EP without DP. DP+EP (all2all dispatch through
    the EP group) is not emulated.
    """
    if os.environ.get("VLLM_FAKE_EP", "0") not in ("1", "true", "True"):
        return False
    if get_fake_tp_size() < 2:
        raise ValueError("VLLM_FAKE_EP=1 needs VLLM_FAKE_TP >= 2: vLLM only enables "
                         "expert parallelism when the TP world size is above 1")
    return True

def apply_fake_tp_patches(tp_size: int) -> None:
    """
    Patch vLLM internals to simulate tensor parallelism on a single GPU.

    Patches:
    1. parallel_state — world size and rank accessors, and the TP group, which
       becomes an emulated GroupCoordinator (communication recorded, not run)
    2. ModelConfig head counts — sharded by the fake TP size

    Args:
        tp_size: The simulated tensor parallel world size (e.g. 2, 4, 8).
    """
    import vllm.distributed.parallel_state as ps
    from vllm import distributed
    from vllm.model_executor.layers import linear
    from vllm.distributed import communication_op

    # --- Fake TP group ---------------------------------------------------
    # A real vLLM GroupCoordinator for tp_size ranks (this process is rank 0),
    # built lazily by the engine's own init_model_parallel_group on emulated
    # transports (comm_emulation): the engine's own code computes the outputs
    # of every communication it makes on the group (all_reduce, all_gather,
    # all_gatherv, reduce_scatter(v), gather, send/recv, dispatch/combine, ...),
    # and each data-moving call is recorded as a COMM range with its tainted
    # inputs and bound arguments.
    from doolyprof.profiler.utils.comm_emulation import build_emulated_group
    emulated_groups = {}

    def get_fake_tp_group():
        group = emulated_groups.get("tp")
        if group is None:
            real = getattr(ps, "_TP", None)
            prefix = real.unique_name.split(":")[0] if real is not None else "tp"
            group = emulated_groups["tp"] = build_emulated_group(tp_size, prefix)
        return group

    # --- parallel_state --------------------------------------------------
    # TP group functions
    ps.get_tp_group = get_fake_tp_group
    communication_op.get_tp_group = get_fake_tp_group
    ps.get_tensor_model_parallel_world_size = lambda: tp_size
    ps.get_tensor_model_parallel_rank = lambda: 0

    # Pipeline parallel functions (use defaults for single-GPU)
    ps.get_pipeline_model_parallel_world_size = lambda: 1
    ps.get_pipeline_model_parallel_rank = lambda: 0

    # Data parallel functions (use defaults for single-GPU)
    ps.get_data_parallel_world_size = lambda: 1
    ps.get_data_parallel_rank = lambda: 0

    # Context parallel functions (use defaults)
    ps.get_prefill_context_model_parallel_world_size = lambda: 1
    ps.get_prefill_context_model_parallel_rank = lambda: 0
    ps.get_decode_context_model_parallel_world_size = lambda: 1
    ps.get_decode_context_model_parallel_rank = lambda: 0
    
    try:
        from vllm.config.model import ModelConfig
        # Store original method
        original_get_num_attention_heads = ModelConfig.get_num_attention_heads

        def fake_get_num_attention_heads(self, parallel_config):
            """Patched version that uses fake TP size instead of real parallel_config."""
            num_heads = self.model_arch_config.total_num_attention_heads
            result = num_heads // tp_size  # Use fake TP size instead of parallel_config.tensor_parallel_size
            print(f"[FakeTP] get_num_attention_heads: {num_heads} // {tp_size} = {result} (vs original: {num_heads // parallel_config.tensor_parallel_size})")
            return result

        # Apply the patch
        ModelConfig.get_num_attention_heads = fake_get_num_attention_heads

        print(f"[FakeTP] Patched ModelConfig.get_num_attention_heads to use TP={tp_size}")

    except ImportError:
        print(f"[FakeTP] Warning: Could not patch ModelConfig.get_num_attention_heads")
    
    try:
        from vllm.config.model import ModelConfig
        # Store original method
        original_get_num_kv_heads = ModelConfig.get_num_kv_heads

        def fake_get_num_kv_heads(self, parallel_config):
            """Patched version that uses fake TP size instead of real parallel_config.

            Mirrors vLLM's own semantics: MLA collapses to one KV head during decode,
            and KV heads are replicated rather than split when the total is smaller
            than the TP degree, so every rank keeps at least one.

            Needed because some backends read the KV head count from ModelConfig
            rather than from the attention layer. TritonAttentionMetadataBuilder
            computes seq_threshold_3D = MIN_LAUNCH_GRID_SIZE_2D // num_heads_kv, so an
            unsharded count makes that threshold too small and selects the 2D kernel
            at batch sizes where a real TP rank would select the 3D kernel.
            """
            if self.use_mla:
                # When using MLA during decode it becomes MQA
                return 1
            total_num_kv_heads = self.get_total_num_kv_heads()
            result = max(1, total_num_kv_heads // tp_size)  # fake TP size, not parallel_config
            print(f"[FakeTP] get_num_kv_heads: {total_num_kv_heads} // {tp_size} = {result} (vs original: {max(1, total_num_kv_heads // parallel_config.tensor_parallel_size)})")
            return result

        # Apply the patch
        ModelConfig.get_num_kv_heads = fake_get_num_kv_heads

        print(f"[FakeTP] Patched ModelConfig.get_num_kv_heads to use TP={tp_size}")

    except ImportError:
        print(f"[FakeTP] Warning: Could not patch ModelConfig.get_num_kv_heads")


    # MIGHT NOT NEED THIS ANYMORE:
    # # --- distributed communication (no-ops) ------------------------------
    # distributed.tensor_model_parallel_all_reduce = lambda x: x
    # distributed.tensor_model_parallel_all_gather = lambda x: x

    # # --- linear layers ---------------------------------------------------
    # linear.get_tensor_model_parallel_world_size = lambda: tp_size
    # linear.get_tensor_model_parallel_rank = lambda: 0


    print(f"[FakeTP] Patches applied: TP={tp_size}")
