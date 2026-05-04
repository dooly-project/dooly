"""
Fake Tensor Parallelism for SGLang single-GPU profiling.

Mirrors `dooly/profiler/utils/fake_tp.py` (which targets vLLM). Patches
SGLang's `parallel_state` so that parallel layers (RowParallelLinear,
ColumnParallelLinear, VocabParallelEmbedding) see world_size = `tp_size`
and shard their weights as if running under TP > 1, while actual
execution stays on a single GPU.

IMPORTANT: `apply_fake_tp_patches_sglang()` must run **inside the SGLang
scheduler subprocess** (i.e., `tainted_run_scheduler_process` in
`dooly/taintracer/sglang_subprocess.py`), BEFORE `run_scheduler_process`
is invoked. Python globals are per-process, so patching from the parent
process has no effect on the child.

Usage (called from `tainted_run_scheduler_process`):

    fake_tp = int(os.environ.get("DOOLY_FAKE_TP", "1"))
    if fake_tp > 1:
        from dooly.sglang.profiler.utils.fake_tp_sglang import apply_fake_tp_patches_sglang
        apply_fake_tp_patches_sglang(fake_tp)
"""

import os

import torch
from torch.profiler import record_function


def get_fake_tp_size() -> int:
    """Read the fake TP size from the DOOLY_FAKE_TP env var (default 1)."""
    return int(os.environ.get("DOOLY_FAKE_TP", "1"))


def _format_tensor_shape(tensor) -> str:
    """Format a tensor's shape for `COMM:` trace annotations, preserving taint
    metadata when present. Mirrors `dooly.profiler.utils.fake_tp.format_tensor_shape`
    so SGLang traces look the same as vLLM traces."""
    try:
        from dooly.sglang.taintracer.tensor import TaintedTensor
        if isinstance(tensor, TaintedTensor):
            return tensor.taint_str_with_history()
    except Exception:
        pass
    if isinstance(tensor, torch.Tensor):
        dims = [f"?({d})" for d in tensor.shape]
        return "[" + ", ".join(dims) + "]"
    return None


def _comm_annotation(op_name: str, *tensor_args) -> str:
    """Build a `COMM: {op_name} IN:[<shape1> | <shape2>]` annotation string."""
    shapes = [s for s in (_format_tensor_shape(t) for t in tensor_args) if s]
    body = " | ".join(shapes) if shapes else "no_inputs"
    return f"COMM: {op_name} IN:[{body}]"


def apply_fake_tp_patches_sglang(tp_size: int) -> None:
    """Patch SGLang internals to simulate TP > 1 on a single GPU.

    Patches `sglang.srt.distributed.parallel_state`:
      - `get_tp_group()` returns a `FakeTPGroup` with `world_size=tp_size`
      - `get_tensor_model_parallel_world_size()` returns `tp_size`
      - `get_tensor_model_parallel_rank()` returns `0`
      - Same for the attention TP variants (`_attn_*`).

    The real `init_distributed_environment(world_size=1)` should still
    run beforehand so that low-level torch.distributed init succeeds with
    a single rank; layers then read the *patched* TP accessors and shard
    their weights for `tp_size` ranks, materializing only rank 0's slice.

    Args:
        tp_size: Simulated tensor-parallel world size (e.g. 2, 4, 8).
    """
    import sglang.srt.distributed.parallel_state as ps

    class FakeTPGroup:
        """Stand-in for SGLang's `GroupCoordinator` at a fake `world_size`.

        Layer init reads `.world_size` and `.rank_in_group` for shard math.
        At runtime, communication methods are no-ops (or shape-preserving
        fakes) since there is only one real rank.
        """

        world_size = tp_size
        rank_in_group = 0
        local_rank = 0
        rank = 0
        ranks = [0]
        local_size = 1
        device = torch.device("cuda")
        cpu_group = None
        device_group = None
        next_rank = 0
        prev_rank = 0

        def all_reduce(self, x):
            # With a single real rank, sum-across-ranks ≡ x. Wrap in
            # `record_function` so the would-be collective shows up in the
            # trace as `COMM: all_reduce IN:[<input shape with taint>]`,
            # matching the vLLM FakeTP convention.
            with record_function(_comm_annotation("all_reduce", x)):
                return x

        def all_gather(self, x, dim=0):
            # Replicate to keep downstream shapes consistent for profiling.
            with record_function(_comm_annotation("all_gather", x)):
                if isinstance(x, torch.Tensor):
                    return torch.cat([x] * self.world_size, dim=dim)
                return x

        def reduce_scatter(self, x, dim=0, world_size=None, group_name=None):
            # Take the rank-0 slice; preserves shape contract.
            with record_function(_comm_annotation("reduce_scatter", x)):
                if isinstance(x, torch.Tensor):
                    ws = world_size if world_size is not None else self.world_size
                    return list(x.chunk(ws, dim=dim))[0]
                return x

        def gather(self, x, dst=0, dim=-1):
            with record_function(_comm_annotation("gather", x)):
                return x

        def broadcast(self, x, src=0):
            with record_function(_comm_annotation("broadcast", x)):
                return x

        def broadcast_tensor_dict(self, td=None, src=0, group=None, metadata_group=None):
            with record_function(_comm_annotation("broadcast_tensor_dict")):
                return td if td is not None else {}

        def __getattr__(self, name):
            # Catch-all so unknown method calls (rare SGLang paths) don't crash;
            # we return a no-op that passes through the first positional arg.
            print(f"[FakeTP-SGLang] WARN: unknown attr {name!r} on FakeTPGroup; passthrough no-op")
            def _passthrough(*args, **kwargs):
                return args[0] if args else None
            return _passthrough

    fake_group = FakeTPGroup()

    # Regular TP
    ps.get_tp_group = lambda: fake_group
    ps.get_tensor_model_parallel_world_size = lambda: tp_size
    ps.get_tensor_model_parallel_rank = lambda: 0

    # Attention TP (SGLang has a separate _ATTN_TP group; if user-set
    # attn_tp matches tp this is straightforward, otherwise the caller
    # would need to plumb a separate attn_tp_size).
    if hasattr(ps, "get_attn_tp_group"):
        ps.get_attn_tp_group = lambda: fake_group
    if hasattr(ps, "get_attn_tensor_model_parallel_world_size"):
        ps.get_attn_tensor_model_parallel_world_size = lambda: tp_size
    if hasattr(ps, "get_attn_tensor_model_parallel_rank"):
        ps.get_attn_tensor_model_parallel_rank = lambda: 0

    # `srt/distributed/communication_op.py` does `from .parallel_state import
    # get_tp_group, get_attn_tp_group` at module-load and caches the function
    # references locally. Reassigning `parallel_state.get_tp_group` doesn't
    # update those cached bindings, so the standalone helpers
    # (`tensor_model_parallel_all_reduce` etc.) would still call the original
    # group. Patch the cached references too. (Same trick vLLM's fake_tp uses.)
    try:
        import sglang.srt.distributed.communication_op as comm_op
        if hasattr(comm_op, "get_tp_group"):
            comm_op.get_tp_group = lambda: fake_group
        if hasattr(comm_op, "get_attn_tp_group"):
            comm_op.get_attn_tp_group = lambda: fake_group
    except ImportError:
        pass

    # `srt/layers/dp_attention.py` imports the attn-TP accessors at module
    # load and exposes `get_attention_tp_size()` (and friends) as wrappers
    # over those cached references. The FLASHINFER attention metadata reads
    # `num_q_heads = total_heads // get_attention_tp_size()` (see
    # `srt/layers/attention/flashinfer_backend.py:925`). Without patching
    # these cached refs, the FLASHINFER kernels see the **unsharded** head
    # count even though weights are sharded — silently wrong.
    try:
        import sglang.srt.layers.dp_attention as dpa
        if hasattr(dpa, "get_attn_tensor_model_parallel_world_size"):
            dpa.get_attn_tensor_model_parallel_world_size = lambda: tp_size
        if hasattr(dpa, "get_attn_tensor_model_parallel_rank"):
            dpa.get_attn_tensor_model_parallel_rank = lambda: 0
        if hasattr(dpa, "get_attn_tp_group"):
            dpa.get_attn_tp_group = lambda: fake_group
        # Also the regular TP refs that dp_attention caches.
        if hasattr(dpa, "get_tensor_model_parallel_world_size"):
            dpa.get_tensor_model_parallel_world_size = lambda: tp_size
        if hasattr(dpa, "get_tensor_model_parallel_rank"):
            dpa.get_tensor_model_parallel_rank = lambda: 0
        if hasattr(dpa, "get_tp_group"):
            dpa.get_tp_group = lambda: fake_group
    except ImportError:
        pass

    # Wrap FlashInferIndicesUpdaterPrefill / Decode init so we can introspect
    # the *actually stored* `self.num_qo_heads` / `self.num_kv_heads` after
    # init — these are the values FlashInfer kernels read at runtime. If our
    # dp_attention patches are working, the stored values should be sharded.
    # FlashInferAttnBackend itself doesn't store these; it only computes them
    # inline and forwards to the indices updaters, which DO store them.
    try:
        from sglang.srt.layers.attention import flashinfer_backend as _fib

        for _cls_name in ("FlashInferIndicesUpdaterPrefill", "FlashInferIndicesUpdaterDecode"):
            _cls = getattr(_fib, _cls_name, None)
            if _cls is None or getattr(_cls, "_dooly_faketp_wrapped", False):
                continue
            _orig_init = _cls.__init__
            _name_for_closure = _cls_name

            def _make_wrapped(orig, name):
                def _wrapped(self, *a, **kw):
                    orig(self, *a, **kw)
                    print(
                        f"[FakeTP-SGLang] {name} stored attrs: "
                        f"num_qo_heads={getattr(self, 'num_qo_heads', '?')}, "
                        f"num_kv_heads={getattr(self, 'num_kv_heads', '?')}, "
                        f"head_dim={getattr(self, 'head_dim', '?')}, "
                        f"data_type={getattr(self, 'data_type', '?')}",
                        flush=True,
                    )
                return _wrapped

            _cls.__init__ = _make_wrapped(_orig_init, _name_for_closure)
            _cls._dooly_faketp_wrapped = True
    except ImportError:
        pass

    print(f"[FakeTP-SGLang] Patches applied: TP={tp_size}", flush=True)
