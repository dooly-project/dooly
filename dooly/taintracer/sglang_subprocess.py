"""
Wrapper for sglang's ``run_scheduler_process`` that re-installs taint
tracking hooks inside the spawned subprocess.

sglang always uses ``mp.set_start_method("spawn")``, so the child process
starts from a clean Python interpreter — all monkey-patches applied in the
parent (``patch_autoconfig``, ``install_hooks``, ``patch_int_in_modules``,
``DummyModelLoader`` patch) are lost.  This module provides a drop-in
replacement that re-applies every patch before the real scheduler code runs.

Usage in ``run-tracer-sglang.py``::

    from dooly.taintracer.sglang_subprocess import (
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

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")


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
    from dooly.taintracer import (
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
        from dooly.taintracer.registry import TAINT_REGISTRY
        dump_path = f"{dump_base}.{tp_rank}"
        while True:
            time.sleep(2)
            with open(dump_path, "wb") as f:
                pickle.dump(dict(TAINT_REGISTRY), f)
            print(f"[taintracer] Dumped {len(TAINT_REGISTRY)} entries to {dump_path}")

    print("[DEBUG] threadingprocess running")
    dump_thread = threading.Thread(target=dump_to_file, daemon=True)
    dump_thread.start()


    # ------------------------------------------------------------------
    # 4. Delegate to the real scheduler, then dump registry on exit.
    # ------------------------------------------------------------------
    from sglang.srt.managers.scheduler import run_scheduler_process

    print("[DEBUG] run process")
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