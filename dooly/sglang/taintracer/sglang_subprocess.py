"""
Wrapper for sglang's ``run_scheduler_process`` that re-installs taint
tracking hooks inside the spawned subprocess.

sglang always uses ``mp.set_start_method("spawn")``, so the child process
starts from a clean Python interpreter — all monkey-patches applied in the
parent (``patch_autoconfig``, ``install_hooks``, ``patch_int_in_modules``,
``DummyModelLoader`` patch) are lost.  This module provides a drop-in
replacement that re-applies every patch before the real scheduler code runs.

Usage in ``run-tracer-sglang.py``::

    from dooly.sglang.taintracer.sglang_subprocess import (
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
import pdb
import threading
from .registry import register_taint
from .types import TaintedInt, TaintedFloat, Taint
from .tensor import TaintedTensor
import traceback
import torch

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

# Attributes to taint in SchedulerConfig
SERVER_ARGS_TAINT_MAP = {
    # Note: Commenting out max_num_batched_tokens to prevent buffer tainting
    # Only runtime values (total_num_scheduled_tokens) should be tainted as NUM_TOKS
    'max_total_num_tokens': 'MAX_NUM_TOKS',
    'max_running_requests': 'MAX_NUM_REQS',

}

_original_server_args_getattribute = None
_original_input_batch_num_reqs = None

def patch_server_args(server_args):
    """Patch SchedulerConfig and InputBatch for workload tainting.

    1. SchedulerConfig: max_num_batched_tokens -> NUM_TOKS, max_num_seqs -> NUM_REQS
       These taint the buffer sizes used in tensor creation.

    2. InputBatch.num_reqs: Returns TaintedInt for the ACTUAL runtime batch size.
       This is needed because num_reqs = len(req_id_to_index) is a plain int,
       and we need the runtime batch dimension to be tainted for LogitsProcessor etc.

    """
    server_args_cls  = type(server_args)
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
                    register_taint(raw, taint_name)
                    return TaintedInt(raw, taint_name)
                return value

            Scheduler.__getattribute__ = patched_getattribute
    except ImportError as e:
        print(f"[DEBUG @ patch_input_buffers] SchedulerConfig ImportError - {e}", flush=True)

    # Patch SchedulerOutput to taint num_scheduled_tokens and total_num_scheduled_tokens
    try:
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
                    register_taint(value, Taint('NUM_TOKS'))
                    return TaintedInt(value, 'NUM_TOKS')
            if name == 'input_ids':
                if isinstance(value, torch.Tensor) and not isinstance(value, TaintedTensor):
                    register_taint(len(value), Taint('NUM_TOKS'))
                    return TaintedTensor(value, [Taint('NUM_TOKS')])
            return value

        ScheduleBatch.__getattribute__ = patched_scheduler_output_getattribute

        from sglang.srt.model_executor.forward_batch_info import ForwardBatch                                                                                                                         
                                                                                                                                                                                                
        _original_init_new = ForwardBatch.init_new.__func__                                                                                                                                           
                                                                                                                                                                                                        
        @classmethod                                              
        def patched_init_new(cls, batch, model_runner):
            ret = _original_init_new(cls, batch, model_runner)                                                                                                                                        
            num_reqs = len(batch.seq_lens)
            register_taint(num_reqs, Taint('NUM_REQS'))   
            if not isinstance(ret.seq_lens, TaintedTensor):                                                                                                                                            
                ret.seq_lens = TaintedTensor(ret.seq_lens, [Taint('NUM_REQS')])
            if not isinstance(ret.positions, TaintedTensor):                                                                                                                         
                ret.positions = TaintedTensor(ret.positions, [Taint('NUM_REQS')])                                                                                                
            return ret                                                                                                                                                                           
        
        ForwardBatch.init_new = patched_init_new  


        print("[DEBUG @ patch_input_buffers] SchedulerOutput patch applied", flush=True)
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
    from dooly.sglang.taintracer import (
        install_hooks,
        patch_autoconfig,
        patch_int_in_modules,
    )

    # patch_autoconfig(verbose=True)

    # ------------------------------------------------------------------
    # 2. Re-install tensor factory hooks, custom-op hooks, and int()
    #    patches for the rotary-embedding layer.
    # ------------------------------------------------------------------
    # install_hooks(
    #     int_patch_modules=[
    #         "sglang.srt.layers.rotary_embedding",
    #     ]
    # )

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
        from dooly.sglang.taintracer.registry import TAINT_REGISTRY
        dump_path = f"{dump_base}.{tp_rank}"
        while True:
            time.sleep(2)
            with open(dump_path, "wb") as f:
                pickle.dump(dict(TAINT_REGISTRY), f)

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
    install_hooks()

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
