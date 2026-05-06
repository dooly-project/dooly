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


def find_free_port() -> int:
    """Find a free port by binding to port 0 and letting the OS assign one."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


# Add project root to path for dooly imports
# run-tracer.py -> run/ -> dooly/ -> dooly (repo root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Parse --gpu early BEFORE importing torch / CUDA so the requested physical
# GPU is remapped to local cuda:0 inside this process and any subprocesses.
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--gpu", type=str, default="0")
_pre_args, _ = _pre_parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_pre_args.gpu)

import torch
import torch.profiler
os.environ["SGLANG_ENABLE_TORCH_INFERENCE_MODE"] = "1"

# Parse full arguments after the CUDA device mask is established.
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B", help="Model name or path")
parser.add_argument("--tp", "--tensor_parallel_size", type=int, default=1, dest="tensor_parallel_size", help="Fake tensor parallel size (weights sharded but run on 1 GPU)")
parser.add_argument("--dtype", type=str, default="bfloat16", help="Model dtype")
parser.add_argument("--prompts", type=str, default="Hello, how are you? I am fine, and ", help="Prompt to run")
parser.add_argument("--max-tokens", type=int, default=1, help="Max tokens to generate")
parser.add_argument("--trace_dir", type=str, default="/home/jkim38/dooly/profiler/vllm_traces_tp", help="Directory to save trace")
parser.add_argument("--gpu", type=str, default="0", help="GPU device index to use")
parser.add_argument("--attention-backend", type=str, default="fa3", help="Attention backend for SGLang. Accepts SGLang names like flashinfer/fa3/fa4/triton, plus legacy aliases like FLASHINFER/FLASH_ATTN/TRITON_ATTN.")
parser.add_argument("--flash-attn-version", type=int, default=2, choices=[2, 3, 4], help="Only used when --attention-backend is a FlashAttention alias. Maps to SGLang fa3/fa4.")
parser.add_argument("--quantization", type=str, default=None, choices=["fp8", "awq", "gptq"], help="Optional quantization scheme for SGLang dummy loading.")
parser.add_argument("--tokenizer", type=str, default=None, help="Override tokenizer (useful for models with broken tokenizer configs)")
parser.add_argument("--enable-torch-compile", action="store_true", help="Follow SGLang main-branch torch.compile path.")
parser.add_argument("--torch-compile-max-bs", type=int, default=32, help="Maximum batch size to compile when --enable-torch-compile is set.")
parser.add_argument("--context-length", type=int, default=None, help="Override model context length. Lower this to reduce memory.")
parser.add_argument("--max-running-requests", type=int, default=263, help="Max running requests for the SGLang engine.")
parser.add_argument("--max-total-tokens", type=int, default=521, help="Max total tokens for the SGLang engine.")
parser.add_argument("--mem-fraction-static", type=float, default=None, help="Optional mem_fraction_static override for the SGLang engine.")
args = parser.parse_args()

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    print(
        f"[Tracer] physical --gpu={args.gpu} mapped via "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"to local cuda:{torch.cuda.current_device()}"
    )

_SGLANG_SUPPORTED_BACKENDS = {
    "triton",
    "torch_native",
    "flex_attention",
    "nsa",
    "cutlass_mla",
    "fa3",
    "fa4",
    "flashinfer",
    "flashmla",
    "trtllm_mla",
    "trtllm_mha",
    "dual_chunk_flash_attn",
    "aiter",
    "wave",
    "intel_amx",
    "ascend",
    "intel_xpu",
}


def _normalize_attention_backend(raw_backend: str, flash_attn_version: int) -> str:
    normalized = raw_backend.strip()
    upper = normalized.upper()
    lower = normalized.lower()

    if upper == "FLASHINFER":
        return "flashinfer"
    if upper == "TRITON_ATTN":
        return "triton"
    if upper == "FLASH_ATTN" or lower == "flash_attn":
        return "fa4" if flash_attn_version == 4 else "fa3"

    if lower in _SGLANG_SUPPORTED_BACKENDS:
        return lower

    raise ValueError(
        f"Unsupported SGLang attention backend '{raw_backend}'. "
        f"Use one of: {sorted(_SGLANG_SUPPORTED_BACKENDS)} "
        "or legacy aliases FLASHINFER/FLASH_ATTN/TRITON_ATTN."
    )


sglang_attn_backend = _normalize_attention_backend(
    args.attention_backend,
    args.flash_attn_version,
)
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

# Patch torch.nn.Parameter.__new__ so that when constructed from a
# TaintedTensor, the data is unwrapped to a plain torch.Tensor before
# delegating. This preserves the Parameter subclass identity (e.g.
# SGLang's parameter subclasses), which carries
# attributes like `weight_loader` via @property — those are unreachable
# if the resulting object is downgraded to a plain TaintedTensor by
# Parameter.__new__'s alt-path (data.detach().requires_grad_(rg)).
_original_parameter_new = torch.nn.Parameter.__new__

def _patched_parameter_new(cls, data=None, requires_grad=True):
    from dooly.taintracer.tensor import TaintedTensor
    if isinstance(data, TaintedTensor):
        # Unwrap: get the plain Tensor view of the same storage, so that
        # type(data) is torch.Tensor and Parameter.__new__ takes the
        # _make_subclass(cls, ...) path that actually returns `cls`.
        with torch._C._DisableTorchDispatch():
            data = torch.Tensor._make_subclass(torch.Tensor, data, False)
    return _original_parameter_new(cls, data=data, requires_grad=requires_grad)

torch.nn.Parameter.__new__ = _patched_parameter_new

# Patch SGLang's DummyModelLoader for taint-traced dummy loads.
# The upstream SGLang dummy path initializes the model, runs quant-method
# postprocessing, then fills weights with random values. Under taint hooks,
# some modules can capture CPU as `_load_device`, which breaks CUDA-only
# quant paths; and some dummy-initialized tensors can remain on CPU.
try:
    import sglang.srt.model_loader.loader as sglang_loader

    DummyModelLoader = sglang_loader.DummyModelLoader
    original_dummy_load_model = DummyModelLoader.load_model

    def patched_dummy_load_model(self, *, model_config, device_config):
        if sglang_loader.get_bool_env_var("SGL_CPU_QUANTIZATION"):
            return sglang_loader.load_model_with_cpu_quantization(
                self,
                model_config=model_config,
                device_config=device_config,
            )

        quant_config = sglang_loader._get_quantization_config(
            model_config,
            self.load_config,
        )
        target_device = torch.device(device_config.device)

        with sglang_loader.set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = sglang_loader._initialize_model(
                    model_config,
                    self.load_config,
                    quant_config,
                )

            if torch.cuda.is_available() and target_device.type == "cuda":
                for module in model.modules():
                    if hasattr(module, "_load_device"):
                        module._load_device = target_device

            for _, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is None:
                    continue
                if (
                    hasattr(module, "is_weights_quantized")
                    and module.is_weights_quantized()
                ):
                    continue
                quant_method.process_weights_after_loading(module)

            sglang_loader.initialize_dummy_weights(model)

            if torch.cuda.is_available() and target_device.type == "cuda":
                def _convert(t):
                    if t.is_meta or t.device == target_device:
                        return t
                    return t.to(target_device)

                model._apply(_convert)

            sglang_loader.post_load_weights(model, model_config)

        return model.eval()

    DummyModelLoader.load_model = patched_dummy_load_model
except ImportError:
    print("[DebugPatch] Could not import SGLang DummyModelLoader")

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
    if args.context_length is not None:
        max_model_len = min(args.context_length, max_model_len)
        print(f"[Tracer] Overriding context_length to {max_model_len}")

    try:
    #     # Apply debug patches before LLM init
    #     try:
    #         patch_for_debugging()
    #     except:
    #         pass

        from dooly.taintracer.sglang_subprocess import (
            tainted_run_scheduler_process, TaintedEngine
        )

        # print(Engine.run_scheduler_process_func.__qualname__)

        # Set TORCH_TRACE_DIR for sglang_subprocess.py to pick up inside the spawned process.
        os.environ["TORCH_TRACE_DIR"] = args.trace_dir

        prompts = [args.prompts] * 5
        engine_kwargs = dict(
            model_path=args.model,
            tokenizer_path=args.tokenizer if args.tokenizer else args.model,
            dtype=args.dtype,
            attention_backend=sglang_attn_backend,
            load_format="dummy",
            trust_remote_code=True,
            max_running_requests=args.max_running_requests,
            max_total_tokens=args.max_total_tokens,
            context_length=max_model_len,
            enable_metrics=True,
            enable_tokenizer_batch_encode=True,
        )
        if args.mem_fraction_static is not None:
            engine_kwargs["mem_fraction_static"] = args.mem_fraction_static
        if args.quantization:
            engine_kwargs["quantization"] = args.quantization
        if args.enable_torch_compile:
            engine_kwargs["enable_torch_compile"] = True
            engine_kwargs["torch_compile_max_bs"] = args.torch_compile_max_bs
        engine_kwargs["disable_cuda_graph"] = True
        engine_kwargs["disable_piecewise_cuda_graph"] = True

        llm = TaintedEngine(**engine_kwargs)
        llm.start_profile(output_dir=args.trace_dir, activities=[
                "CPU",
                "GPU",
            ], with_stack=False, record_shapes=True, )
        sampling_params={"max_new_tokens": 0, "temperature": 0.0}
        outputs = llm.generate(prompt=prompts, sampling_params=sampling_params)
        llm.stop_profile()

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
                _json.dump(_data, f, indent=2)
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
