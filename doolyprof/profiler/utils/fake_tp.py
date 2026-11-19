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


def get_fake_dp_size() -> int:
    """Return the fake data-parallel size from the VLLM_FAKE_DP environment variable."""
    return int(os.environ.get("VLLM_FAKE_DP", "1"))


def get_fake_ep() -> bool:
    """Whether expert parallelism is on (VLLM_FAKE_EP=1) over the fake ranks.

    vLLM's MoE layer derives its parallel layout from the TP and DP world sizes
    and ``ParallelConfig.enable_expert_parallel`` (``FusedMoEParallelConfig.make``):
    with EP on, ``ep_size = dp_size * tp_size``, every rank holds
    ``num_experts / ep_size`` whole experts and the expert weights are not
    TP-sharded. Because both world sizes are the patched fake ones, passing
    ``enable_expert_parallel=True`` to the engine is all fake EP needs: rank 0
    of the deployment is built and run on one GPU.

    * TP+EP (no DP): the MoE output is all-reduced over the TP group, which fake
      TP records as a COMM all_reduce, exactly as vLLM does.
    * DP+EP (VLLM_FAKE_DP > 1): tokens are dispatched to the experts' ranks and
      combined back through the EP group's all2all manager, which on the default
      backend gathers over the DP group (all_gatherv) and reduce-scatters back
      (reduce_scatterv); every DP rank is assumed to run a batch like this one.
    """
    if os.environ.get("VLLM_FAKE_EP", "0") not in ("1", "true", "True"):
        return False
    if get_fake_tp_size() * get_fake_dp_size() < 2:
        raise ValueError("VLLM_FAKE_EP=1 needs VLLM_FAKE_TP * VLLM_FAKE_DP >= 2: vLLM only enables "
                         "expert parallelism over more than one rank")
    return True

def apply_fake_tp_patches(tp_size: int, dp_size: int | None = None) -> None:
    """
    Patch vLLM internals to simulate tensor (and data) parallelism on a single GPU.

    Patches:
    1. parallel_state — world size and rank accessors, and the TP group, which
       becomes an emulated GroupCoordinator (communication recorded, not run)
    2. ModelConfig head counts — sharded by the fake TP size
    3. with dp_size > 1: emulated DP and EP groups, and a forward context in
       which every DP rank runs as many tokens as this one (``_apply_fake_dp``)

    Args:
        tp_size: The simulated tensor parallel world size (e.g. 1, 2, 4, 8).
        dp_size: The simulated data parallel world size (default: VLLM_FAKE_DP).
    """
    if dp_size is None:
        dp_size = get_fake_dp_size()
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
    # TP group functions (a TP size of 1 keeps the engine's own group)
    if tp_size > 1:
        ps.get_tp_group = get_fake_tp_group
        communication_op.get_tp_group = get_fake_tp_group
        ps.get_tensor_model_parallel_world_size = lambda: tp_size
        ps.get_tensor_model_parallel_rank = lambda: 0

    # Pipeline parallel functions (use defaults for single-GPU)
    ps.get_pipeline_model_parallel_world_size = lambda: 1
    ps.get_pipeline_model_parallel_rank = lambda: 0

    # --- the deployment's parallel configuration --------------------------
    # The engine runs one process with its own configuration (TP 1, DP 1); one
    # rank of the fake deployment sees TP = tp_size and DP = dp_size, as DP rank
    # 0. The model is built against this view, the forward context is set with
    # it, and hybrid models size their recurrent state from it: engine code that
    # reads the sizes from parallel_config rather than from the parallel groups
    # (vLLM's ParallelConfig.use_sequence_parallel_moe, the TP size an MLA
    # layer picks its prefill kernel by) then decides as on a real rank.
    import copy

    def fake_parallel_vllm_config(vllm_config):
        view = copy.copy(vllm_config)
        parallel = copy.copy(vllm_config.parallel_config)
        object.__setattr__(parallel, "tensor_parallel_size", tp_size)
        if dp_size > 1:
            object.__setattr__(parallel, "data_parallel_size", dp_size)
            object.__setattr__(parallel, "data_parallel_rank", 0)
        object.__setattr__(view, "parallel_config", parallel)
        return view

    # Data parallel functions
    ps.get_data_parallel_world_size = lambda: dp_size
    ps.get_data_parallel_rank = lambda: 0
    if dp_size > 1:
        _apply_fake_dp(tp_size, dp_size, emulated_groups, fake_parallel_vllm_config)

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


    # --- hybrid models' recurrent-state shape -----------------------------
    # The engine sizes the KV cache of a hybrid model (Mamba / Gated DeltaNet /
    # short-conv layers next to attention) at config time from the model class's
    # get_mamba_state_shape_from_config(vllm_config), which reads the per-rank
    # size from parallel_config.tensor_parallel_size: 1 here, because the engine
    # runs as one process. Attention is sized from the patched head counts above
    # and the layers themselves are built at the fake TP size, so the state was
    # sized as unsharded: the attention block came out twice as large as on a
    # real TP rank, and when the two page sizes happened to match exactly
    # (Qwen3.6-35B-A3B on vLLM 0.17.1) no padding was recorded and the KV-cache
    # setup failed. Every resolved model class answers with the fake TP size.
    try:
        import functools
        from vllm.model_executor.models.registry import _ModelRegistry

        def wrap_init(model_cls):
            """Build the model against the deployment's parallel configuration."""
            original = model_cls.__init__
            if getattr(original, "_fake_parallel", False):
                return

            @functools.wraps(original)
            def __init__(self, *args, **kwargs):
                if kwargs.get("vllm_config") is None:
                    return original(self, *args, **kwargs)
                from vllm.config import set_current_vllm_config
                kwargs["vllm_config"] = fake_parallel_vllm_config(kwargs["vllm_config"])
                # layers read the configuration being built from the current one
                with set_current_vllm_config(kwargs["vllm_config"]):
                    original(self, *args, **kwargs)
            __init__._fake_parallel = True
            model_cls.__init__ = __init__

        def wrap_state_shape(model_cls):
            import inspect
            try:
                raw = inspect.getattr_static(model_cls, "get_mamba_state_shape_from_config")
            except AttributeError:
                return
            if not isinstance(raw, classmethod) or getattr(raw.__func__, "_fake_tp", False):
                return
            original = raw.__func__

            def get_mamba_state_shape_from_config(cls, vllm_config, *args, **kwargs):
                return original(cls, fake_parallel_vllm_config(vllm_config), *args, **kwargs)
            get_mamba_state_shape_from_config._fake_tp = True
            # on the class that defines it, so subclasses keep their own cls
            owner = next(c for c in model_cls.__mro__ if "get_mamba_state_shape_from_config" in c.__dict__)
            owner.get_mamba_state_shape_from_config = classmethod(get_mamba_state_shape_from_config)

        original_resolve = _ModelRegistry.resolve_model_cls

        def resolve_model_cls(self, *args, **kwargs):
            resolved = original_resolve(self, *args, **kwargs)
            model_cls = resolved[0] if isinstance(resolved, tuple) else resolved
            if isinstance(model_cls, type):
                wrap_state_shape(model_cls)
                wrap_init(model_cls)
            return resolved

        _ModelRegistry.resolve_model_cls = resolve_model_cls
    except ImportError:
        print("[FakeTP] Warning: could not patch hybrid models' state-shape sizing")

    # MIGHT NOT NEED THIS ANYMORE:
    # # --- distributed communication (no-ops) ------------------------------
    # distributed.tensor_model_parallel_all_reduce = lambda x: x
    # distributed.tensor_model_parallel_all_gather = lambda x: x

    # # --- linear layers ---------------------------------------------------
    # linear.get_tensor_model_parallel_world_size = lambda: tp_size
    # linear.get_tensor_model_parallel_rank = lambda: 0


    print(f"[FakeTP] Patches applied: TP={tp_size}, DP={dp_size}")


def _rebind(module_attr_owner, name, original, replacement):
    """Point every already-imported binding of ``original`` (``from x import name``)
    at ``replacement``, so modules imported before the patch see it too."""
    import sys
    setattr(module_attr_owner, name, replacement)
    for mod in list(sys.modules.values()):
        try:
            if getattr(mod, name, None) is original:
                setattr(mod, name, replacement)
        except Exception:  # noqa: BLE001 - lazy / proxy modules
            continue


def _apply_fake_dp(tp_size: int, dp_size: int, emulated_groups: dict, fake_dp_vllm_config) -> None:
    """Fake data parallelism: this process is DP rank 0 of ``dp_size``.

    The engine runs one DP rank (its own config says data_parallel_size=1), but
    the model sees what a DP deployment gives it:

    * ``get_dp_group()``: an emulated group of ``dp_size`` ranks, and
      ``get_ep_group()``: an emulated expert-parallel group of
      ``dp_size * tp_size`` ranks with the engine's all2all manager. The MoE
      layer sizes itself from these (``ep_size = dp_size * tp_size``) and, with
      expert parallelism, dispatches/combines through the EP group. Data
      received on them is this rank's own, so gathered expert ids are valid.
    * the forward context: built for a ``dp_size``-rank deployment in which every
      rank runs as many tokens as this one, by the engine's own
      ``set_forward_context`` (its per-rank token counts are what the all2all
      manager sizes dispatch and combine with).
    """
    import copy
    import inspect
    import vllm.distributed.parallel_state as ps
    import vllm.forward_context as fc
    from doolyprof.profiler.utils.comm_emulation import build_emulated_group

    # The engine's configuration (as the DP view the forward context gets): a
    # group first asked for during a forward (vLLM 0.17.1's MoE runner) is built
    # outside any config scope, where the engine's communicator and all2all
    # manager cannot read the backend and expert-parallel settings.
    engine_config = {}

    def in_engine_config(build):
        from vllm.config import get_current_vllm_config_or_none, set_current_vllm_config
        if get_current_vllm_config_or_none() is None and engine_config.get("config") is not None:
            with set_current_vllm_config(engine_config["config"]):
                return build()
        return build()

    def get_fake_dp_group():
        group = emulated_groups.get("dp")
        if group is None:
            group = emulated_groups["dp"] = in_engine_config(
                lambda: build_emulated_group(dp_size, "dp", replicate=True))
        return group

    def get_fake_ep_group():
        group = emulated_groups.get("ep")
        if group is None:
            group = emulated_groups["ep"] = in_engine_config(lambda: build_emulated_group(
                dp_size * tp_size, "ep", use_all2all=get_fake_ep(), replicate=True))
        return group

    _rebind(ps, "get_dp_group", ps.get_dp_group, get_fake_dp_group)
    _rebind(ps, "get_ep_group", ps.get_ep_group, get_fake_ep_group)

    import contextlib
    import functools
    original_set_forward_context = fc.set_forward_context
    sig = inspect.signature(original_set_forward_context)

    # functools.wraps: callers that adapt their keywords to the installed vLLM
    # (vllm_compat.call_with_supported_kwargs) inspect this signature and must
    # see the engine's own
    @contextlib.contextmanager
    @functools.wraps(original_set_forward_context)
    def set_forward_context(*args, **kwargs):
        bound = sig.bind_partial(*args, **kwargs)
        arguments = bound.arguments
        num_tokens = None
        if arguments.get("vllm_config") is not None:
            arguments["vllm_config"] = fake_dp_vllm_config(arguments["vllm_config"])
            engine_config["config"] = arguments["vllm_config"]
            num_tokens = arguments.get("num_tokens")
            if num_tokens is not None and arguments.get("num_tokens_across_dp") is None:
                arguments["num_tokens_across_dp"] = torch.full((dp_size,), int(num_tokens), dtype=torch.int32)
        with original_set_forward_context(*bound.args, **bound.kwargs) as value:
            dp_metadata = getattr(fc.get_forward_context(), "dp_metadata", None)
            if dp_metadata is not None and num_tokens is not None:
                # Every rank's count is this rank's token count; keep it as that
                # value (with its taint), not the plain int DPMetadata derives
                # from its CPU tensor, so the per-rank sizes a dispatch records
                # follow the token sweep in the comm profiler.
                try:
                    dp_metadata._fake_dp_tokens = num_tokens
                except (AttributeError, TypeError):  # slotted / frozen in some version
                    object.__setattr__(dp_metadata, "_fake_dp_tokens", num_tokens)
            yield value

    _rebind(fc, "set_forward_context", original_set_forward_context, set_forward_context)

    original_chunk_sizes = getattr(fc.DPMetadata, "get_chunk_sizes_across_dp_rank", None)
    if original_chunk_sizes is not None and not getattr(original_chunk_sizes, "_fake_dp", False):
        def get_chunk_sizes_across_dp_rank(self, *args, **kwargs):
            sizes = original_chunk_sizes(self, *args, **kwargs)
            tokens = getattr(self, "_fake_dp_tokens", None)
            if sizes is None or tokens is None:
                return sizes
            # Sequence-parallel MoE (TP x DP + EP) splits every DP rank's tokens
            # over the ranks that list covers: len(sizes) / dp_size per DP rank,
            # ceil(tokens / that) each. Derived from the tainted count, so the
            # per-rank sizes keep its taint.
            per_dp = len(sizes) // dp_size if len(sizes) % dp_size == 0 else 1
            split = (tokens + per_dp - 1) // per_dp if per_dp > 1 else tokens
            return [split if int(s) == int(split) else s for s in sizes]
        get_chunk_sizes_across_dp_rank._fake_dp = True
        fc.DPMetadata.get_chunk_sizes_across_dp_rank = get_chunk_sizes_across_dp_rank
    print(f"[FakeTP] Fake DP={dp_size}: emulated DP group and EP group of {dp_size * tp_size} ranks")
