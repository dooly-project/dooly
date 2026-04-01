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
    from dooly.profiler.utils.fake_tp import apply_fake_tp_patches, get_fake_tp_size
"""

import os
import torch


def get_fake_tp_size() -> int:
    """Return the fake TP size from the VLLM_FAKE_TP environment variable."""
    return int(os.environ.get("VLLM_FAKE_TP", "1"))


def apply_fake_tp_patches(tp_size: int) -> None:
    """
    Patch vLLM internals to simulate tensor parallelism on a single GPU.

    Patches three areas:
    1. parallel_state — world size, rank, and TP group
    2. distributed — all-reduce/all-gather become no-ops
    3. linear layers — world size and rank for column/row parallel

    Args:
        tp_size: The simulated tensor parallel world size (e.g. 2, 4, 8).
    """
    import vllm.distributed.parallel_state as ps
    from vllm import distributed
    from vllm.model_executor.layers import linear

    # --- Fake TP group ---------------------------------------------------
    class FakeTPGroup:
        world_size = tp_size
        rank_in_group = 0
        local_rank = 0
        rank = 0
        device = torch.device("cuda")
        cpu_group = None
        device_group = None

        def all_reduce(self, x):
            return ps.all_reduce_fake(
                tensor=x, group_name="fake_reduce"
            )

        def all_gather(self, x, dim=-1):
            return ps.all_gather_fake(
                tensor=x, dim=dim,
                world_size=self.world_size,
                group_name="fake_gather",
            )

        def reduce_scatter(self, tensor, dim, world_size, group_name):
            return ps.reduce_scatter_fake(
                tensor=tensor, dim=dim,
                world_size=world_size,
                group_name=group_name,
            )

        def patched_fused_scaled_matmul_reduce_scatter(self, x, y, z):
            return ps.patched_fused_scaled_matmul_reduce_scatter_fake(
                tensor=x, tensor2=y, tensor3=z,
                world_size=self.world_size,
                group_name="fake_fused_scaled_matmul_reduce_scatter",
            )

    fake_group = FakeTPGroup()

    # --- parallel_state --------------------------------------------------
    ps.get_tp_group = lambda: fake_group
    ps.get_tensor_model_parallel_world_size = lambda: tp_size
    ps.get_tensor_model_parallel_rank = lambda: 0

    # --- distributed communication (no-ops) ------------------------------
    distributed.tensor_model_parallel_all_reduce = lambda x: x
    distributed.tensor_model_parallel_all_gather = lambda x: x

    # --- linear layers ---------------------------------------------------
    linear.get_tensor_model_parallel_world_size = lambda: tp_size
    linear.get_tensor_model_parallel_rank = lambda: 0

    print(f"[FakeTP] Patches applied: TP={tp_size}")
