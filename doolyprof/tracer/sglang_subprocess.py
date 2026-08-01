"""
Wrapper for SGLang's ``run_scheduler_process`` that re-installs taint
tracking hooks inside the spawned subprocess.

sglang always uses ``mp.set_start_method("spawn")``, so the child process
starts from a clean Python interpreter — all monkey-patches applied in the
parent (``patch_autoconfig``, ``install_hooks``, ``patch_int_in_modules``,
``DummyModelLoader`` patch) are lost.  This module provides a drop-in
replacement that re-applies every patch before the real scheduler code runs.

Usage in ``run-tracer-sglang.py``::

    from doolyprof.tracer.sglang_subprocess import (
        tainted_run_scheduler_process,
    )
    from sglang import Engine

    Engine.run_scheduler_process_func = staticmethod(
        tainted_run_scheduler_process,
    )
    engine = Engine(...)
"""

import os
import time
import threading
from .registry import register_taint
from .types import DimTaint, TaintedInt, TaintedFloat, Taint
from .tensor import TaintedTensor
import traceback

if os.environ.get("DOOLY_ALLOW_TORCH_COMPILE") != "1":
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

# Attributes to taint in SchedulerConfig
SERVER_ARGS_TAINT_MAP = {
    # Note: Commenting out max_num_batched_tokens to prevent buffer tainting
    # Only runtime values (total_num_scheduled_tokens) should be tainted as NUM_TOKS
    'max_total_num_tokens': 'MAX_NUM_TOKS',
    'max_running_requests': 'MAX_NUM_REQS',

}

# Attributes to taint on ModelRunner. SGLang reads runtime caps off of
# ModelRunner (not Scheduler) when sizing internal buffers; without these
# the value is a plain int and the propagation chain breaks at the source.
#   max_total_num_tokens : KV cache pool size on dim 0 (MHATokenToKVPool)
#                          -> store_cache's [?(522)] becomes [MAX_NUM_TOKS(522)].
#   max_running_requests : ReqToTokenPool's req_to_token shape[0]
#                          -> FA3's [?(263), MODEL_CONFIG(131076)] becomes
#                          [MAX_NUM_REQS(263), MODEL_CONFIG(131076)].
MODEL_RUNNER_TAINT_MAP = {
    'max_total_num_tokens': 'MAX_NUM_TOKS',
    'max_running_requests': 'MAX_NUM_REQS',
}

_original_server_args_getattribute = None
_original_model_runner_getattribute = None
_original_eager_fb_view = None
_original_piecewise_replay_prepare = None
_original_input_batch_num_reqs = None
_original_req_to_token_pool_init = None


def _try_register_runtime_dim(value, taint_name):
    """Best-effort shape hint for SGLang runtime helper tensors.

    The registry is value-based, so a runtime value can legitimately collide
    with a model-config value. Leave the existing entry in place on conflict;
    validation should use non-colliding runtime sizes and fail explicitly if a
    trace still labels runtime values as model config.
    """
    if isinstance(value, bool):
        return
    try:
        raw = int(value)
    except (TypeError, ValueError):
        return
    try:
        register_taint(raw, Taint(taint_name))
    except ValueError:
        # Runtime hints should improve trace labels, not change execution or
        # guess through value collisions.
        pass


def _retag_runtime_tensor(value, taint_name, dim=-1):
    """Make one SGLang runtime-boundary tensor dimension authoritative."""
    import torch

    if not isinstance(value, torch.Tensor):
        return value
    if not isinstance(value, TaintedTensor):
        value = TaintedTensor(value, [None] * value.ndim)
    target_dim = dim if dim >= 0 else value.ndim + dim
    if target_dim < 0 or target_dim >= value.ndim:
        raise IndexError(
            f"runtime taint dimension {dim} is invalid for rank {value.ndim}"
        )
    dim_taints = list(value._dim_taints)
    for index, size in enumerate(value.shape):
        if index != target_dim and int(size) == 1:
            dim_taints[index] = None
    dim_taints[target_dim] = DimTaint.from_taint(Taint(taint_name))
    object.__setattr__(value, "_dim_taints", tuple(dim_taints))
    return value


def _retag_forward_batch_runtime_tensors(forward_batch):
    """Restore workload roles after SGLang copies a ForwardBatch into buffers."""
    forward_batch.input_ids = _retag_runtime_tensor(
        forward_batch.input_ids, "NUM_TOKS"
    )
    forward_batch.positions = _retag_runtime_tensor(
        forward_batch.positions, "NUM_TOKS"
    )
    forward_batch.out_cache_loc = _retag_runtime_tensor(
        forward_batch.out_cache_loc, "NUM_TOKS"
    )
    forward_batch.seq_lens = _retag_runtime_tensor(
        forward_batch.seq_lens, "NUM_REQS"
    )
    return forward_batch


def patch_server_args(server_args):
    """Patch SchedulerConfig and InputBatch for workload tainting.

    1. SchedulerConfig: max_num_batched_tokens -> NUM_TOKS, max_num_seqs -> NUM_REQS
       These taint the buffer sizes used in tensor creation.

    2. InputBatch.num_reqs: Returns TaintedInt for the ACTUAL runtime batch size.
       This is needed because num_reqs = len(req_id_to_index) is a plain int,
       and we need the runtime batch dimension to be tainted for LogitsProcessor etc.

    """
    import os
    global _original_server_args_getattribute
    global _original_input_batch_num_reqs

    # print(f"[DEBUG @ patch_input_buffers] Called! PID={os.getpid()}", flush=True)

    # Patch SchedulerConfig
    try:
        from sglang.srt.managers.scheduler import Scheduler
        # print("[DEBUG @ patch_input_buffers] Patching SchedulerConfig", flush=True)

        if _original_server_args_getattribute is None:
            _original_server_args_getattribute = Scheduler.__getattribute__

            def patched_getattribute(self, name):
                value = _original_server_args_getattribute(self, name)
                if name in SERVER_ARGS_TAINT_MAP and isinstance(value, int):
                    raw = int(value)  # strip any existing taint from propagation
                    taint_name = SERVER_ARGS_TAINT_MAP[name]
                    _try_register_runtime_dim(raw, taint_name)
                    return TaintedInt(raw, taint_name)
                return value

            Scheduler.__getattribute__ = patched_getattribute
    except ImportError as e:
        print(f"[DEBUG @ patch_input_buffers] SchedulerConfig ImportError - {e}", flush=True)

    # Patch ModelRunner so reads of max_total_num_tokens return a TaintedInt.
    # The KV cache pool reads it from ModelRunner (not Scheduler), so without
    # this patch the existing tensor factory hooks never see a tainted size and
    # the K/V buffers come out untainted on dim 0.
    try:
        from sglang.srt.model_executor.model_runner import ModelRunner
        global _original_model_runner_getattribute
        global _original_eager_fb_view
        if _original_model_runner_getattribute is None:
            _original_model_runner_getattribute = ModelRunner.__getattribute__

            def patched_model_runner_getattribute(self, name):
                value = _original_model_runner_getattribute(self, name)
                if name in MODEL_RUNNER_TAINT_MAP and isinstance(value, int) and not isinstance(value, TaintedInt):
                    raw = int(value)
                    taint_name = MODEL_RUNNER_TAINT_MAP[name]
                    _try_register_runtime_dim(raw, taint_name)
                    return TaintedInt(raw, taint_name)
                return value

            ModelRunner.__getattribute__ = patched_model_runner_getattribute
        if _original_eager_fb_view is None:
            _original_eager_fb_view = ModelRunner._eager_fb_view

            def patched_eager_fb_view(self, forward_batch, pp_proxy_tensors=None):
                ret = _original_eager_fb_view(
                    self,
                    forward_batch,
                    pp_proxy_tensors=pp_proxy_tensors,
                )
                return _retag_forward_batch_runtime_tensors(ret)

            ModelRunner._eager_fb_view = patched_eager_fb_view
    except ImportError as e:
        print(f"[DEBUG @ patch_input_buffers] ModelRunner ImportError - {e}", flush=True)

    try:
        from sglang.srt.model_executor.piecewise_cuda_graph_runner import (
            PiecewiseCudaGraphRunner,
        )
        global _original_piecewise_replay_prepare
        if _original_piecewise_replay_prepare is None:
            _original_piecewise_replay_prepare = (
                PiecewiseCudaGraphRunner.replay_prepare
            )

            def patched_piecewise_replay_prepare(self, forward_batch, **kwargs):
                ret = _original_piecewise_replay_prepare(
                    self,
                    forward_batch,
                    **kwargs,
                )
                return _retag_forward_batch_runtime_tensors(ret)

            PiecewiseCudaGraphRunner.replay_prepare = (
                patched_piecewise_replay_prepare
            )
    except ImportError as e:
        print(
            f"[DEBUG @ patch_input_buffers] PiecewiseCudaGraphRunner "
            f"ImportError - {e}",
            flush=True,
        )

    try:
        from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
        global _original_req_to_token_pool_init
        if _original_req_to_token_pool_init is None:
            _original_req_to_token_pool_init = ReqToTokenPool.__init__

            def patched_req_to_token_pool_init(
                self,
                size,
                max_context_len,
                device,
                enable_memory_saver,
            ):
                _try_register_runtime_dim(max_context_len, 'MODEL_CONFIG')
                return _original_req_to_token_pool_init(
                    self,
                    size,
                    max_context_len,
                    device,
                    enable_memory_saver,
                )

            ReqToTokenPool.__init__ = patched_req_to_token_pool_init
    except ImportError as e:
        print(f"[DEBUG @ patch_input_buffers] ReqToTokenPool ImportError - {e}", flush=True)

    # Patch SchedulerOutput to taint num_scheduled_tokens and total_num_scheduled_tokens
    try:
        import torch
        from sglang.srt.managers.schedule_batch import ScheduleBatch
        # print("[DEBUG @ patch_input_buffers] Patching SchedulerOutput", flush=True)

        _original_schedule_batch_getattribute = ScheduleBatch.__getattribute__

        def patched_scheduler_output_getattribute(self, name):
            value = _original_schedule_batch_getattribute(self, name)
            # if name == 'extend_lens':
            #     print(f"[DEBUG @ patch_input_buffers] extend_lens accessed!")
            #     if isinstance(value, list) and not isinstance(value, TaintedList):
            #         print(f"[DEBUG @ patch_input_buffers] Registering NUM_REQS")
            #         register_taint(len(value), Taint('NUM_REQS'))
            #         return TaintedList(value)
            if name == 'extend_num_tokens':
                if isinstance(value, int) and not isinstance(value, TaintedInt):
                    _try_register_runtime_dim(value, 'NUM_TOKS')
                    return TaintedInt(value, 'NUM_TOKS')
            # out_cache_loc is computed at runtime (torch.cat of allocator
            # results); its length equals NUM_TOKS but it has no upstream
            # tainted scalar to propagate from, so wrap on read.
            if name == 'out_cache_loc':
                return _retag_runtime_tensor(value, 'NUM_TOKS')
            return value

        ScheduleBatch.__getattribute__ = patched_scheduler_output_getattribute

        from sglang.srt.model_executor.forward_batch_info import ForwardBatch                                                                                                                         
                                                                                                                                                                                                
        _original_init_new = ForwardBatch.init_new.__func__                                                                                                                                           
                                                                                                                                                                                                        
        @classmethod
        def patched_init_new(cls, batch, model_runner):
            ret = _original_init_new(cls, batch, model_runner)
            num_reqs = len(batch.seq_lens)
            _try_register_runtime_dim(num_reqs, 'NUM_REQS')
            _try_register_runtime_dim(num_reqs + 1, 'NUM_REQS')
            for extend_len in getattr(batch, 'extend_lens', None) or ():
                _try_register_runtime_dim(extend_len, 'NUM_TOKS')
            try:
                _try_register_runtime_dim(
                    model_runner.req_to_token_pool.max_context_len,
                    'MODEL_CONFIG',
                )
            except AttributeError:
                pass
            _retag_forward_batch_runtime_tensors(ret)
            # batch_size is set as `len(batch.seq_lens)` at init_new — a plain
            # int. Re-tainting here so `forward_batch.batch_size + 1`
            # propagates to torch.arange(0, batch_size+1) for cu_seqlens_q/k.
            if isinstance(ret.batch_size, int) and not isinstance(ret.batch_size, TaintedInt):
                _try_register_runtime_dim(ret.batch_size, 'NUM_REQS')
                ret.batch_size = TaintedInt(ret.batch_size, 'NUM_REQS')
            # input_ids is the model's entry-point tensor; its length equals
            # the number of scheduled tokens this step. Without this wrap,
            # VocabParallelEmbedding / LlamaModel see [?(N)] on their input.
            if isinstance(ret.input_ids, torch.Tensor):
                num_toks = ret.input_ids.shape[0]
                _try_register_runtime_dim(num_toks, 'NUM_TOKS')
            return ret
        
        ForwardBatch.init_new = patched_init_new  


    except ImportError as e:
        print(f"[DEBUG @ patch_input_buffers] SchedulerOutput ImportError - {e}", flush=True)



def tainted_run_scheduler_process(
    server_args,
    port_args,
    gpu_id,
    tp_rank,
    attn_cp_rank,
    moe_dp_rank,
    moe_ep_rank,
    pp_rank,
    dp_rank,
    pipe_writer,
):
    """Install taint hooks in the subprocess, then run the real scheduler."""

    # ------------------------------------------------------------------
    # 1. Patch AutoConfig so model configs loaded here are tainted.
    # ------------------------------------------------------------------
    from doolyprof.tracer import (
        install_hooks,
        patch_autoconfig,
        patch_int_in_modules,
    )

    # patch_autoconfig(verbose=True)

    # ------------------------------------------------------------------
    # 2. Re-install tensor factory hooks, custom-op hooks, and int()
    #    patches for the rotary-embedding layer.
    # ------------------------------------------------------------------
    # Rebind `int` -> _preserving_int in every module under sglang.srt.* via
    # an import hook. SGLang's model files (sglang.srt.models.*) and rope
    # submodules (sglang.srt.layers.rotary_embedding.*) are imported lazily
    # by the model loader, so a static list misses anything not pre-listed.
    # The prefix hook fires on each import and catches whatever gets pulled
    # in, model-agnostically.
    install_hooks(
        int_patch_prefixes=[
            "sglang.srt.",
        ]
    )

    # ------------------------------------------------------------------
    # 2b. Apply Fake-TP patches if requested. Must run inside the
    #     subprocess (Python globals are per-process) and BEFORE
    #     run_scheduler_process initializes layers that read
    #     parallel_state's TP accessors.
    # ------------------------------------------------------------------
    _fake_tp = int(os.environ.get("DOOLY_FAKE_TP", "1"))
    if _fake_tp > 1:
        from doolyprof.profiler.utils.fake_tp_sglang import apply_fake_tp_patches_sglang
        apply_fake_tp_patches_sglang(_fake_tp)

    # ------------------------------------------------------------------
    # 3. Patch DummyModelLoader.load_model so that:
    #    a) The model-specific module is discovered at load time and its
    #       int() is patched (e.g. sglang.srt.models.llama does
    #       ``int(partial_rotary_factor * self.head_dim)``).
    #    b) Dummy-loaded models are moved to CUDA.
    # ------------------------------------------------------------------
    try:
        import torch
        from sglang.srt.model_loader.loader import DummyModelLoader

        _original_load_model = DummyModelLoader.load_model

        # def _patched_load_model(self, model_config, device_config):
        #     try:
        #         from sglang.srt.model_loader.loader import get_model_architecture

        #         model_cls, _ = get_model_architecture(model_config)
                
        #         # Patch all sglang modules in the inheritance hierarchy (e.g. Llama for Mistral)
        #         modules_to_patch = set()
        #         for cls in model_cls.mro():
        #              if cls.__module__ and cls.__module__.startswith("sglang."):
        #                  modules_to_patch.add(cls.__module__)

        #         patch_int_in_modules(list(modules_to_patch))
        #         print(f"[DebugPatch] Patched int() in DummyModelLoader.load_model for {modules_to_patch}")
        #     except Exception:
        #         print("[DebugPatch] Could not patch int() in DummyModelLoader.load_model")
        #         pass

        #     model = _original_load_model(
        #         self, model_config=model_config, device_config=device_config,
        #     )
        #     if torch.cuda.is_available():
        #         model.to("cuda")
        #     return model

        # DummyModelLoader.load_model = _patched_load_model
    except ImportError:
        pass


    # ------------------------------------------------------------------
    # 3. Run a thread every 2 seconds to dump TAINT_REGISTRY to a file
    # ------------------------------------------------------------------

    dump_base = os.environ.get("TAINT_REGISTRY_DUMP_BASE")

    def dump_to_file():
        if not dump_base:
            return
        import pickle
        from doolyprof.tracer.registry import TAINT_REGISTRY
        dump_path = f"{dump_base}.{tp_rank}"
        tmp_path = dump_path + ".tmp"
        while True:
            time.sleep(2)
            # Write to a temp file then atomically rename. Without this,
            # `open(dump_path, "wb")` truncates first; if the subprocess
            # gets sigterm before pickle.dump finishes, the parent reads a
            # 0-byte file ("Ran out of input"). os.replace is atomic on
            # POSIX, so the consumer either sees the previous valid dump
            # or the new complete one — never a partial.
            try:
                with open(tmp_path, "wb") as f:
                    pickle.dump(dict(TAINT_REGISTRY), f)
                os.replace(tmp_path, dump_path)
            except Exception:
                # On crash mid-write, leave the previous good dump intact.
                pass

    dump_thread = threading.Thread(target=dump_to_file, daemon=True)
    dump_thread.start()


    from sglang.srt.managers.scheduler import run_scheduler_process
    try:
        patch_server_args(server_args)
    except Exception as e:
        print(f"[tainted_run_scheduler_process] patch_server_args failed: {e}", flush=True)
        traceback.print_exc()


    # ------------------------------------------------------------------
    # 4. Delegate to the real scheduler wrapped in torch profiler so that
    #    GPU kernels (running in this subprocess) are captured in the trace.
    # ------------------------------------------------------------------
    # install_hooks()

    try:
        return run_scheduler_process(
            server_args,
            port_args,
            gpu_id,
            tp_rank,
            attn_cp_rank,
            moe_dp_rank,
            moe_ep_rank,
            pp_rank,
            dp_rank,
            pipe_writer,
        )
    except:
        return

from sglang.srt.entrypoints.engine import Engine

class TaintedEngine(Engine):
    run_scheduler_process_func = staticmethod(tainted_run_scheduler_process)
