"""
Single-process TP=N profiling: Load sharded weights on 1 GPU, profile compute.

This patches vLLM's parallel state so linear layers see tp_size=N while
running in a single process. Weights are sharded but no actual NCCL is used.

Usage:
    python test.py --model meta-llama/Llama-3.1-8B --tp 2 --trace_dir ./traces
"""
import os
import pdb
import sys
import socket
import argparse
import time
import torch


def find_free_port() -> int:
    """Find a free port by binding to port 0 and letting the OS assign one."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


# Add project root to path for dooly imports
# run-tracer.py -> run/ -> dooly/ -> dooly (repo root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Parse arguments BEFORE any heavy imports
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B", help="Model name or path")
parser.add_argument("--tp", "--tensor_parallel_size", type=int, default=1, dest="tensor_parallel_size", help="Fake tensor parallel size (weights sharded but run on 1 GPU)")
parser.add_argument("--dtype", type=str, default="bfloat16", help="Model dtype")
parser.add_argument("--prompts", type=str, default="Hello, how are you? I am fine, and ", help="Prompt to run")
parser.add_argument("--max-tokens", type=int, default=1, help="Max tokens to generate")
parser.add_argument("--trace_dir", type=str, default="/home/jkim38/dooly/profiler/vllm_traces_tp", help="Directory to save trace")
parser.add_argument("--gpu", type=int, default=0, help="GPU device index to use")
# Accept either SGLang's native names ("flashinfer", "fa3", "fa4", "triton",
# "torch_native", ...) or the legacy vLLM-style names (FLASHINFER, FLASH_ATTN,
# TRITON_ATTN). Mapped below before passing to ServerArgs.
parser.add_argument("--attention-backend", type=str, default="flashinfer",
                    help="Attention backend. SGLang names: flashinfer, fa3, fa4, "
                         "triton, torch_native. Legacy aliases: FLASHINFER, "
                         "FLASH_ATTN (->fa3), TRITON_ATTN (->triton).")
parser.add_argument("--tokenizer", type=str, default=None, help="Override tokenizer (useful for models with broken tokenizer configs)")
args = parser.parse_args()

import torch
import torch.profiler
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ["SGLANG_ENABLE_TORCH_INFERENCE_MODE"] = "1"

_BACKEND_ALIASES = {
    "FLASHINFER": "flashinfer",
    "FLASH_ATTN": "fa3",
    "TRITON_ATTN": "triton",
}
sglang_attn_backend = _BACKEND_ALIASES.get(args.attention_backend, args.attention_backend)
print(f"[Tracer] SGLang attention_backend={sglang_attn_backend} "
      f"(from --attention-backend={args.attention_backend})")

FAKE_TP = args.tensor_parallel_size
# DOOLY_FAKE_TP is read inside `tainted_run_scheduler_process` (subprocess)
# to apply parallel_state patches; SGLang's real engine still inits as a
# single-rank ProcessGroup.
os.environ["DOOLY_FAKE_TP"] = str(FAKE_TP)

# Install taint hooks BEFORE any vLLM imports
# This patches AutoConfig.from_pretrained to return TaintedConfig objects
# and installs hooks on tensor factory functions for taint propagation
import tempfile
from dooly.taintracer.hooks import install_hooks
from dooly.taintracer.config_sglang import patch_autoconfig, patch_input_buffers
from dooly.taintracer.registry import TAINT_REGISTRY
from dooly.taintracer import disable_taint_tracking, enable_taint_tracking

# Set up registry dump path so the subprocess can write it back to us
_registry_dump_file = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
_registry_dump_file.close()
if "TAINT_REGISTRY_DUMP_BASE" not in os.environ:
    print(f"dump base: {_registry_dump_file.name}")
    os.environ["TAINT_REGISTRY_DUMP_BASE"] = _registry_dump_file.name

# Disable taint tracking initially - we'll enable it only during the actual generate() call
# This prevents semantic confusion during warm-up/compilation where values are reused incorrectly
# disable_taint_tracking()
# print("[Tracer] Taint tracking DISABLED during initialization/warm-up")

patch_autoconfig()
# patch_common_attention_metadata()
# patch_input_buffers()
install_hooks()

# # Patch DummyModelLoader to force CUDA move
# # (dummy models are loaded into CPU for some reason when tainting is enabled)
# try:
#     from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
#     original_dummy_load_weights = DummyModelLoader.load_weights

#     def patched_dummy_load_weights(self, model, model_config):
#         original_dummy_load_weights(self, model, model_config)
#         if torch.cuda.is_available():
#             model.to("cuda")

#     DummyModelLoader.load_weights = patched_dummy_load_weights
# except ImportError:
#     print("[DebugPatch] Could not import DummyModelLoader")

# # Debug patch: Track num_tokens in _dummy_run
# try:
#     import vllm.v1.worker.gpu_model_runner as gmr_module
#     from dooly.taintracer.types import TaintedInt

#     _orig_dummy_run = None
#     _orig_get_slot_mappings = None

#     def patch_for_debugging():
#         global _orig_dummy_run, _orig_get_slot_mappings
#         if hasattr(gmr_module, 'GPUModelRunner'):
#             GPUModelRunner = gmr_module.GPUModelRunner

#             if _orig_dummy_run is None:
#                 _orig_dummy_run = GPUModelRunner._dummy_run
#                 def debug_dummy_run(self, num_tokens, *args, **kwargs):
#                     return _orig_dummy_run(self, num_tokens, *args, **kwargs)
#                 GPUModelRunner._dummy_run = debug_dummy_run

#             if _orig_get_slot_mappings is None:
#                 _orig_get_slot_mappings = GPUModelRunner._get_slot_mappings
#                 def debug_get_slot_mappings(self, num_tokens_padded, *args, **kwargs):
#                     return _orig_get_slot_mappings(self, num_tokens_padded, *args, **kwargs)
#                 GPUModelRunner._get_slot_mappings = debug_get_slot_mappings

#             print("[DebugPatch] Installed debug wrappers for _dummy_run and _get_slot_mappings", flush=True)

#     # Will patch after GPUModelRunner is defined
# except Exception as e:
#     print(f"[DebugPatch] Could not setup debug patches - {e}", flush=True)

# # Apply fake TP patches using shared utility, then import vLLM
# from dooly.profiler.utils.fake_tp import apply_fake_tp_patches
# if FAKE_TP > 1:
#     apply_fake_tp_patches(FAKE_TP)

# from vllm import LLM, SamplingParams
from transformers import AutoConfig
# from sglang import  Engine

# # Debug: verify multiprocessing mode
# import vllm.envs as envs
# print(f"[DEBUG] VLLM_ENABLE_V1_MULTIPROCESSING = {envs.VLLM_ENABLE_V1_MULTIPROCESSING}", flush=True)

def get_max_model_len(model_name: str, default: int = 4096) -> int:
    """Auto-detect max_model_len from model config."""
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        # Try common attribute names for max position embeddings
        for attr in ["max_position_embeddings", "n_positions", "max_seq_len"]:
            if hasattr(config, attr):
                val = getattr(config, attr)
                if val is not None:
                    print(f"[Tracer] Auto-detected max_model_len={val} from {attr}")
                    return int(val)
        print(f"[Tracer] Could not find max position embeddings, using default={default}")
        return default
    except Exception as e:
        print(f"[Tracer] Error loading config: {e}, using default={default}")
        return default

if __name__ == "__main__":
    print(f"[Tracer] Tracing {args.model} with fake TP={FAKE_TP}")

    # Auto-detect max_model_len from model config, capped at 32768 to avoid OOM
    max_model_len = min(get_max_model_len(args.model), 32768)

    try:
    #     # Apply debug patches before LLM init
    #     try:
    #         patch_for_debugging()
    #     except:
    #         pass

        from dooly.taintracer.sglang_subprocess import (
        tainted_run_scheduler_process, TaintedEngine
        )
        # from sglang import Engine

        # print(Engine.run_scheduler_process_func.__qualname__)

        # Set TORCH_TRACE_DIR for sglang_subprocess.py to pick up inside the spawned process.
        os.environ["TORCH_TRACE_DIR"] = args.trace_dir

        prompts = [args.prompts] * 5
        llm = TaintedEngine(
            model_path=args.model,
            attention_backend=sglang_attn_backend,
            enable_metrics=True,
            disable_cuda_graph=True,
            disable_piecewise_cuda_graph=True,
            load_format="dummy",
            enable_tokenizer_batch_encode=True,
            max_running_requests=263,
            max_total_tokens=521,
        )
        llm.start_profile(output_dir=args.trace_dir, activities=[
                "CPU",
                "GPU",
            ], with_stack=False, record_shapes=True, )
        sampling_params={"max_new_tokens": 0, "temperature": 0.0}
        outputs = llm.generate(prompt=prompts, sampling_params=sampling_params)
        llm.stop_profile()
        print(f"output: {outputs}")

        # Sleep so children process dumping thread has time to write back to file
        time.sleep(5)
        llm.shutdown()

        # Load per-rank registry dumps written by subprocesses
        import pickle
        import glob
        dump_base = os.environ.get("TAINT_REGISTRY_DUMP_BASE")
        if dump_base:
            for dump_path in sorted(glob.glob(f"{dump_base}.*")):
                try:
                    with open(dump_path, "rb") as f:
                        TAINT_REGISTRY.update(pickle.load(f))
                    print(f"[taintracer] Loaded registry from {dump_path}")
                except (EOFError, pickle.UnpicklingError, OSError) as e:
                    # Subprocess can crash mid-dump (esp. on backend init
                    # failures like fa3 on Ampere) and leave a 0-byte or
                    # truncated pickle. Skip rather than aborting the rest
                    # of the post-processing.
                    print(f"[taintracer] WARNING: skipped {dump_path}: {e}")

        # Inject the merged taint registry as a top-level key in the SGLang
        # trace JSON, mirroring `dooly_trace_handler` in run/run-tracer.py.
        # SGLang writes traces as gzipped JSON via start_profile/stop_profile;
        # we round-trip through gzip here. We pick the most recently modified
        # `*.trace.json[.gz]` in args.trace_dir to avoid stale files.
        import json as _json
        import gzip as _gzip
        _trace_candidates = sorted(
            glob.glob(os.path.join(args.trace_dir, "*.trace.json.gz")) +
            glob.glob(os.path.join(args.trace_dir, "*.trace.json")),
            key=os.path.getmtime,
        )
        if _trace_candidates:
            _trace_path = _trace_candidates[-1]
            _is_gz = _trace_path.endswith(".gz")
            with (_gzip.open(_trace_path, "rt") if _is_gz else open(_trace_path, "rt")) as f:
                _data = _json.load(f)
            _data["dooly_taint_registry"] = {
                str(v): str(t) for v, t in TAINT_REGISTRY.items()
                if not isinstance(t, dict)
            }
            _out_path = _trace_path[:-3] if _is_gz else _trace_path
            with open(_out_path, "wt") as f:
                _json.dump(_data, f)
            if _is_gz and _out_path != _trace_path:
                os.remove(_trace_path)
            print(f"[Tracer] Wrote trace + taint registry "
                  f"({len(_data['dooly_taint_registry'])} entries) to {_out_path}")
        else:
            print(f"[Tracer] WARNING: no trace file found in {args.trace_dir}")

        print(f"[Tracer] Done tracing {args.model}, output saved to {args.trace_dir}")

        # Print registry contents
        print("\n" + "="*80)
        print("TAINT_REGISTRY CONTENTS")
        print("="*80)
        print(f"Total entries: {len(TAINT_REGISTRY)}")

        if len(TAINT_REGISTRY) > 0:
            # Separate pure and MIX entries
            pure_entries = {k: v for k, v in TAINT_REGISTRY.items() if not isinstance(v, dict)}
            mix_entries = {k: v for k, v in TAINT_REGISTRY.items() if isinstance(v, dict)}

            if pure_entries:
                print(f"\nPure Taints ({len(pure_entries)} entries):")
                # Sort by value (taint name) then by key
                for key in sorted(pure_entries.keys()):
                    value = pure_entries[key]
                    print(f"  {key:6d} → {value}")

            if mix_entries:
                print(f"\nMIX Taints ({len(mix_entries)} entries):")
                # Sort by key
                for key in sorted(mix_entries.keys()):
                    value = mix_entries[key]
                    components_str = ', '.join(f"{qty}×{taint}" for taint, qty in value['components'].items())
                    print(f"  {key:6d} → MIX [{components_str}]")
        else:
            print("  (Registry is empty)")

        print("="*80 + "\n")
            
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
