"""
Single-process TP=N profiling: Load sharded weights on 1 GPU, profile compute.

This patches vLLM's parallel state so linear layers see tp_size=N while
running in a single process. Weights are sharded but no actual NCCL is used.

Usage:
    python test.py --model meta-llama/Llama-3.1-8B --tp 2 --trace_dir ./traces
"""
import os
import sys
import socket
import argparse


def find_free_port() -> int:
    """Find a free port by binding to port 0 and letting the OS assign one."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


# Add project root to path for doolyprof imports
# run-tracer.py -> run/ -> doolyprof/ -> repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Parse arguments BEFORE any heavy imports
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B", help="Model name or path")
parser.add_argument("--tp", "--tensor_parallel_size", type=int, default=1, dest="tensor_parallel_size", help="Fake tensor parallel size (weights sharded but run on 1 GPU)")
parser.add_argument("--dtype", type=str, default="bfloat16", help="Model dtype")
parser.add_argument("--prompts", type=str, default="Hello, how are you? I am fine, and ", help="Prompt to run")
parser.add_argument("--batch-size", type=int, default=5, help="Requested number of identical prompts in the traced generate() call; raised to the nearest value whose NUM_REQS / cu_seqlens / NUM_TOKS avoid the taint registry (see choose_trace_batch_size)")
parser.add_argument("--max-tokens", type=int, default=1, help="Max tokens to generate")
parser.add_argument("--trace_dir", type=str, default="./vllm_traces_tp", help="Directory to save trace")
parser.add_argument("--gpu", type=int, default=0, help="GPU device index to use")
parser.add_argument("--attention-backend", type=str, default="FLASH_ATTN", help="Attention backend (FLASHINFER, FLASH_ATTN, XFORMERS)")
parser.add_argument("--flash-attn-version", type=int, default=2, choices=[2, 3, 4], help="Pin flash-attention version when --attention-backend=FLASH_ATTN. Without this, vLLM auto-picks FA3 on Hopper / FA4 on Blackwell.")
parser.add_argument("--quantization", type=str, default=None, choices=["fp8", "awq", "gptq"], help="Optional quantization scheme. 'fp8' requires Hopper+ (H100). With load_format=dummy, vLLM initializes random quantized weights — no quantized checkpoint needed.")
parser.add_argument("--tokenizer", type=str, default=None, help="Override tokenizer (useful for models with broken tokenizer configs)")
parser.add_argument("--dp", "--data_parallel_size", type=int, default=1, dest="data_parallel_size", help="Fake data parallel size: this process is DP rank 0, every DP rank is assumed to run the same batch; with --enable-expert-parallel the MoE dispatches/combines across the DP ranks")
parser.add_argument("--enable-expert-parallel", action="store_true", help="Fake EP over the fake TP x DP ranks (vLLM TP+EP, DP+EP); needs --tp * --dp >= 2")
parser.add_argument("--gpu-memory-cap", type=float, default=0.9,
                    help="Largest fraction of the GPU the traced engine may take (what is free, at most this). "
                         "Raise it when the KV cache left after the weights cannot hold a trace batch whose "
                         "request count avoids every config value (the tracer says so).")
parser.add_argument("--kv-cache-dtype", type=str, default="auto", help="vLLM KV-cache dtype literal (auto, fp8, fp8_e4m3, fp8_e5m2, ...); recorded in each attention module's state hash")
args = parser.parse_args()

import torch
import torch.profiler
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

print(f"[Tracer] Set VLLM_ATTENTION_BACKEND={args.attention_backend}")

FAKE_TP = args.tensor_parallel_size
os.environ["VLLM_FAKE_TP"] = str(FAKE_TP)
FAKE_DP = args.data_parallel_size
os.environ["VLLM_FAKE_DP"] = str(FAKE_DP)
os.environ["VLLM_FAKE_EP"] = "1" if args.enable_expert_parallel else "0"

# Install taint hooks BEFORE any vLLM imports
# This patches AutoConfig.from_pretrained to return TaintedConfig objects
# and installs hooks on tensor factory functions for taint propagation
from doolyprof.tracer.hooks import install_hooks
from doolyprof.tracer.config import patch_autoconfig, patch_input_buffers
from doolyprof.tracer.registry import TAINT_REGISTRY, choose_trace_batch
from doolyprof.tracer import disable_taint_tracking, enable_taint_tracking

# Disable taint tracking initially - we'll enable it only during the actual generate() call
# This prevents semantic confusion during warm-up/compilation where values are reused incorrectly
disable_taint_tracking()
print("[Tracer] Taint tracking DISABLED during initialization/warm-up")

# install_hooks() MUST run before patch_autoconfig()/patch_input_buffers():
# it patches torch.compile with the taint-aware wrapper, and patch_input_buffers()
# transitively imports vllm.model_executor.layers.vocab_parallel_embedding (whose
# get_masked_input_and_mask is @torch.compile-decorated at import time). Patching
# torch.compile first ensures that decorator captures the taint-aware wrapper so it
# bypasses Dynamo for TaintedTensor inputs (avoids the tp>1 PatchedSize crash).
install_hooks(
    int_patch_modules=[
        "vllm.model_executor.layers.rotary_embedding",
    ]
)
patch_autoconfig()
# patch_common_attention_metadata()
patch_input_buffers()

# Patch torch.nn.Parameter.__new__ so that when constructed from a
# TaintedTensor, the data is unwrapped to a plain torch.Tensor before
# delegating. This preserves the Parameter subclass identity (e.g.
# vLLM's BasevLLMParameter / ModelWeightParameter), which carries
# attributes like `weight_loader` via @property — those are unreachable
# if the resulting object is downgraded to a plain TaintedTensor by
# Parameter.__new__'s alt-path (data.detach().requires_grad_(rg)).
_original_parameter_new = torch.nn.Parameter.__new__

def _patched_parameter_new(cls, data=None, requires_grad=True, **kwargs):
    from doolyprof.tracer.tensor import TaintedTensor
    # Rebuilding a Parameter from another one's attributes,
    # ``type(p)(p.to(device), **p.__dict__)`` (vLLM's Transformers backend, for
    # SmolLM3 and other models without a native vLLM implementation), passes
    # the labels this patch attached as a keyword: they are the labels again.
    dim_taints = kwargs.pop("_dim_taints", None)
    if kwargs:  # anything else is for the original constructor to accept or reject
        return _original_parameter_new(cls, data=data, requires_grad=requires_grad, **kwargs)
    if isinstance(data, TaintedTensor):
        # Save the dim taints before unwrapping so the weight Parameter keeps
        # its MODEL_CONFIG dims (re-attached below). The unwrap itself preserves
        # the vLLM Parameter subclass identity (BasevLLMParameter/weight_loader).
        dim_taints = tuple(data._dim_taints)
        # Unwrap: get the plain Tensor view of the same storage, so that
        # type(data) is torch.Tensor and Parameter.__new__ takes the
        # _make_subclass(cls, ...) path that actually returns `cls`.
        with torch._C._DisableTorchDispatch():
            data = torch.Tensor._make_subclass(torch.Tensor, data, False)
    param = _original_parameter_new(cls, data=data, requires_grad=requires_grad)
    sizes = tuple(int(s) for s in torch.Tensor.size(param))
    if sizes:
        # A weight's shape is set by the model (and its parallel split), never
        # by the batch: every dim without a label is MODEL_CONFIG. vLLM builds
        # some weights from plain sizes or rebuilds them after loading (Kimi-K3's
        # fused q/kv down-projection, 2112 wide), which left them unlabelled and
        # their outputs ?(2112).
        from doolyprof.tracer.types import DimTaint as _DimTaint, Taint as _Taint
        from doolyprof.tracer.registry import register_taint as _register
        known = list(dim_taints or getattr(data, "_dim_taints", None) or ())
        known += [None] * (len(sizes) - len(known))
        filled = []
        for size, t in zip(sizes, known):
            if (t is None or (isinstance(t, _DimTaint) and t.taint is None)) and size > 1:
                t = _DimTaint(_Taint("MODEL_CONFIG"), merge_history=None)
                if size not in TAINT_REGISTRY:
                    # so the trace batch avoids it (choose_trace_batch) and a
                    # plain tensor of that size reads as the weight's dim
                    _register(size, _Taint("MODEL_CONFIG"))
            filled.append(t)
        dim_taints = tuple(filled)
    if dim_taints is not None:
        object.__setattr__(param, "_dim_taints", dim_taints)
    return param

torch.nn.Parameter.__new__ = _patched_parameter_new


def _parameter_getitem(self, index):
    """A slice of a weight (Kimi-K3 runs its q/kv down-projection on
    ``fused_qkv_a_g_proj.weight[:qkv_a_rows]``) has fixed sizes, like the
    weight. Indexing a plain Parameter bypasses the tracer, so the slice lost
    the weight's labels and the GEMM output came out ?(2112)."""
    out = torch.Tensor.__getitem__(self, index)
    if (isinstance(out, torch.Tensor) and getattr(self, "_dim_taints", None) is not None
            and getattr(out, "_dim_taints", None) is None and out.dim() > 0):
        from doolyprof.tracer.types import (DimTaint as _DimTaint, Taint as _Taint,
                                            TaintedInt as _TaintedInt, split_offset_label as _split)

        def _batch(x):  # a bound that follows the batch makes the slice follow it too
            t = x.taint if isinstance(x, _TaintedInt) else None
            while isinstance(t, _DimTaint):
                t = t.taint
            return t is not None and (_split(t) is not None or str(t) == "MIX")
        bounds = index if isinstance(index, tuple) else (index,)
        if not any(_batch(b) or (isinstance(b, slice) and any(_batch(x) for x in (b.start, b.stop, b.step)))
                   for b in bounds):
            object.__setattr__(out, "_dim_taints", tuple(
                _DimTaint(_Taint("MODEL_CONFIG"), merge_history=None) if s > 1 else None
                for s in torch.Tensor.size(out)))
    return out


torch.nn.Parameter.__getitem__ = _parameter_getitem


# A labelled weight's .shape / .size() as tainted ints while taints are tracked
# (hooks.parameter_size): vLLM's CUTLASS FP8 GEMM on Blackwell sizes its output
# from b.shape[1] of the weight.
from doolyprof.tracer.hooks import parameter_size as _parameter_size  # noqa: E402

torch.nn.Parameter.size = _parameter_size
torch.nn.Parameter.shape = property(_parameter_size)

# Patch DummyModelLoader to force CUDA move
# (dummy models are loaded into CPU for some reason when tainting is enabled)
try:
    from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
    original_dummy_load_weights = DummyModelLoader.load_weights

    def patched_dummy_load_weights(self, model, model_config):
        if torch.cuda.is_available():
            # vLLM's FP8 / quantization paths capture `layer._load_device =
            # torch.get_default_device()` at create_weights time, then later
            # use it inside process_weights_after_loading to allocate the
            # real (non-meta) weight. With our taint hooks active, that
            # captured device can resolve to CPU even though vLLM wraps
            # initialize_model in `with target_device:` — and a CPU weight
            # then fails scaled_fp8_quant (CUDA-only). Pin it to cuda:0
            # before dummy load so vLLM's per-layer materialize+quantize
            # loop runs on CUDA (it frees the BF16 staging tensor after
            # each layer, so this doesn't double-allocate).
            cuda_dev = torch.device("cuda:0")
            for module in model.modules():
                if hasattr(module, "_load_device"):
                    module._load_device = cuda_dev

        original_dummy_load_weights(self, model, model_config)

        if torch.cuda.is_available():
            # Move CPU-placed dummy weights to CUDA (tainting sometimes
            # leaves them on CPU). Leave meta tensors alone — vLLM's
            # quantization init handles them in process_weights_after_loading.
            def _convert(t):
                if t.is_meta or t.device.type == "cuda":
                    return t
                return t.to("cuda")
            model._apply(_convert)

    DummyModelLoader.load_weights = patched_dummy_load_weights
except ImportError:
    print("[DebugPatch] Could not import DummyModelLoader")

# Debug patch: Track num_tokens in _dummy_run
try:
    import vllm.v1.worker.gpu_model_runner as gmr_module
    from doolyprof.tracer.types import TaintedInt

    _orig_dummy_run = None
    _orig_get_slot_mappings = None

    def patch_for_debugging():
        global _orig_dummy_run, _orig_get_slot_mappings
        if hasattr(gmr_module, 'GPUModelRunner'):
            GPUModelRunner = gmr_module.GPUModelRunner

            if _orig_dummy_run is None:
                _orig_dummy_run = GPUModelRunner._dummy_run
                def debug_dummy_run(self, num_tokens, *args, **kwargs):
                    return _orig_dummy_run(self, num_tokens, *args, **kwargs)
                GPUModelRunner._dummy_run = debug_dummy_run

            if _orig_get_slot_mappings is None:
                _orig_get_slot_mappings = GPUModelRunner._get_slot_mappings
                def debug_get_slot_mappings(self, num_tokens_padded, *args, **kwargs):
                    return _orig_get_slot_mappings(self, num_tokens_padded, *args, **kwargs)
                GPUModelRunner._get_slot_mappings = debug_get_slot_mappings

            print("[DebugPatch] Installed debug wrappers for _dummy_run and _get_slot_mappings", flush=True)

    # Will patch after GPUModelRunner is defined
except Exception as e:
    print(f"[DebugPatch] Could not setup debug patches - {e}", flush=True)

# Apply fake TP patches using shared utility, then import vLLM
from doolyprof.profiler.utils.fake_tp import apply_fake_tp_patches
from doolyprof.profiler.utils.gpu_memory import engine_memory_fraction
if FAKE_TP > 1 or FAKE_DP > 1:
    apply_fake_tp_patches(FAKE_TP, FAKE_DP)

# The emulated group sizes are configuration like the model's own: vLLM's
# collectives shape their buffers with the plain-int world size (all_gather:
# output.reshape((world_size,) + input_size)), which otherwise leaves the
# gathered dim untracked. Registered before the engine starts, so the trace
# batch is also chosen clear of them.
from doolyprof.tracer.registry import lookup_taint as _lookup_taint, register_taint as _register_taint
from doolyprof.tracer.types import Taint as _Taint
_ep = FAKE_TP * FAKE_DP if args.enable_expert_parallel else 1
for _size in sorted({FAKE_TP, FAKE_DP, _ep} - {0, 1}):
    if _lookup_taint(_size) is None:
        _register_taint(_size, _Taint('MODEL_CONFIG'))

from vllm import LLM, SamplingParams
from transformers import AutoConfig

# Debug: verify multiprocessing mode
import vllm.envs as envs
print(f"[DEBUG] VLLM_ENABLE_V1_MULTIPROCESSING = {envs.VLLM_ENABLE_V1_MULTIPROCESSING}", flush=True)

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
        # The installed transformers may not know the architecture (LFM2.5's
        # lfm2_moe under transformers 4.57, which vLLM 0.17.1 still serves):
        # read the same fields from the checkpoint's config.json.
        try:
            from transformers import PretrainedConfig
            raw, _ = PretrainedConfig.get_config_dict(model_name, trust_remote_code=True)
            for attr in ["max_position_embeddings", "n_positions", "max_seq_len"]:
                if raw.get(attr) is not None:
                    print(f"[Tracer] Auto-detected max_model_len={raw[attr]} from {attr} in config.json "
                          f"(AutoConfig: {type(e).__name__})")
                    return int(raw[attr])
        except Exception:  # noqa: BLE001
            pass
        print(f"[Tracer] Error loading config: {e}, using default={default}")
        return default

def encode_trace_prompt(tokenizer_name: str, prompt: str) -> list:
    """Token ids of the traced prompt, from the checkpoint's own tokenizer.

    The engine runs without a tokenizer (skip_tokenizer_init): the traced
    generate() takes token ids, and nothing else in the trace needs text.
    AutoTokenizer builds the tokenizer class the checkpoint names; when the
    installed transformers does not have that class (LFM2.5 and North-Mini
    name transformers 5's TokenizersBackend), the checkpoint's tokenizer.json
    is read with the fast-tokenizer base class, which encodes it the same way.
    """
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    except Exception as e:  # noqa: BLE001
        print(f"[Tracer] AutoTokenizer cannot build {tokenizer_name}'s tokenizer ({type(e).__name__}: "
              f"{str(e)[:120]}); reading its tokenizer.json")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_name, trust_remote_code=True)
    return list(tokenizer.encode(prompt))


if __name__ == "__main__":
    print(f"[Tracer] Tracing {args.model} with fake TP={FAKE_TP}")

    # Auto-detect max_model_len from model config, capped at 32768 to avoid OOM
    max_model_len = min(get_max_model_len(args.model), 32768)

    try:
        # Apply debug patches before LLM init
        try:
            patch_for_debugging()
        except:
            pass

        llm_kwargs = dict(
            model=args.model,
            tokenizer=args.tokenizer if args.tokenizer else args.model,
            dtype=args.dtype,
            tensor_parallel_size=1,  # Tell vLLM executor we're single process
            load_format="dummy",
            enforce_eager=True,
            # what is free on the GPU now, at most 0.9 (a fixed 0.9 fails whenever
            # another process holds more than a tenth of the GPU)
            gpu_memory_utilization=engine_memory_fraction(cap=args.gpu_memory_cap, devices=1, who="tracer"),
            attention_backend=args.attention_backend,
            max_model_len=max_model_len,
            trust_remote_code=True,
            max_num_batched_tokens=521,
            max_num_seqs=263,
            kv_cache_dtype=args.kv_cache_dtype,
            # The warm-up generate below runs the very prompts that are then
            # traced; with prefix caching their prefill would be served from
            # cache and vanish from the trace.
            enable_prefix_caching=False,
            # Only the language model is traced. With multimodal inputs on, vLLM
            # also sizes the vision/audio encoders against the scheduler limits
            # and refuses to start when an item exceeds them (Gemma 4: 2496
            # tokens per image > max_num_batched_tokens=521).
            language_model_only=True,
            # The traced prompts are token ids (encode_trace_prompt); the
            # engine needs no tokenizer of its own.
            skip_tokenizer_init=True,
            # Async scheduling (on by default in vLLM 0.33) plans the next step
            # while the current one runs, so how the traced batch is split
            # into steps depends on timing: one Kimi-K3 trace in four had a
            # step with 93 of its 94 requests, 93 is a config value, and the
            # NUM_REQS registration stopped the trace. Lockstep scheduling
            # runs the whole batch in the one step choose_trace_batch sized.
            async_scheduling=False,
        )
        if args.attention_backend == "FLASH_ATTN":
            llm_kwargs["attention_config"] = {"flash_attn_version": args.flash_attn_version}
        if args.quantization:
            llm_kwargs["quantization"] = args.quantization
        from doolyprof.profiler.utils.fake_tp import get_fake_ep
        if get_fake_ep():
            llm_kwargs["enable_expert_parallel"] = True
        llm = LLM(**llm_kwargs)

        # Check weight shapes to verify sharding
        print(f"\n=== Checking weight shapes (should be divided by TP={FAKE_TP}) ===")
        model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        # The V2 model runner (default in vLLM 0.32) samples with a plain class,
        # not an nn.Module; annotate its calls like a module so the sampling
        # ops are attributed and profiled (see hooks.annotate_engine_callable).
        from doolyprof.tracer.hooks import annotate_engine_callable, engine_callables
        for _name, _obj in engine_callables(llm.llm_engine.model_executor.driver_worker.model_runner).items():
            if annotate_engine_callable(_obj):
                print(f"[Tracer] Annotating {type(_obj).__qualname__} calls as MODULE")
        from doolyprof.tracer.hooks import register_module_paths
        print(f"[Tracer] Recorded paths of {register_module_paths(model)} modules for PATH: annotations")
        for name, param in model.named_parameters():
            if "layers.0" in name and ("q_proj" in name or "down_proj" in name):
                print(f"{name}: {param.shape}")

        sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)

        # Pick the trace batch against the registry the engine config filled
        # during init, so the scheduler's NUM_REQS / NUM_TOKS and the derived
        # cu_seqlens length cannot collide with a MODEL_CONFIG value.
        prompt_ids = encode_trace_prompt(args.tokenizer or args.model, args.prompts)
        sched = llm.llm_engine.vllm_config.scheduler_config
        # ... and so the KV cache admits the whole batch in one step (a hybrid
        # model's large blocks can leave room for few requests)
        from doolyprof.tracer.registry import kv_cache_capacity
        cache = llm.llm_engine.vllm_config.cache_config
        kv_capacity = kv_cache_capacity(getattr(cache, "num_gpu_blocks", None),
                                        getattr(cache, "block_size", None), args.max_tokens)
        print(f"[Tracer] KV cache: {getattr(cache, 'num_gpu_blocks', None)} blocks of "
              f"{getattr(cache, 'block_size', None)} tokens"
              + (f", {kv_capacity(len(prompt_ids))} requests of this prompt at once" if kv_capacity else ""))
        # The cache tensors' block count and block size are fixed for the run
        # but in no config the registry read; label them before the trace batch
        # is chosen so its integers avoid them as well.
        from doolyprof.tracer.registry import kv_cache_tensors, register_kv_cache_dims
        _runner = llm.llm_engine.model_executor.driver_worker.model_runner
        _contexts = [getattr(getattr(c, "compilation_config", None), "static_forward_context", None)
                     for c in (getattr(_runner, "vllm_config", None), llm.llm_engine.vllm_config)]
        _cache_tensors = kv_cache_tensors({k: v for c in _contexts if c for k, v in c.items()})
        _cache_dims = register_kv_cache_dims(_cache_tensors)
        print(f"[Tracer] KV-cache tensors: {len(_cache_tensors)}, shapes "
              f"{sorted({tuple(t.shape) for t in _cache_tensors})[:6]}; labelled KV_CACHE: {sorted(_cache_dims)}")
        # weights and what quantization rebuilt from them after loading
        # (repacked weights, scales): fixed sizes
        from doolyprof.tracer.registry import label_module_tensors
        print(f"[Tracer] Labelled the fixed dims of {label_module_tensors(model, skip=_cache_tensors)} "
              f"module tensors MODEL_CONFIG")
        batch_size, prompt_len = choose_trace_batch(
            args.batch_size, len(prompt_ids), sched.max_num_seqs, sched.max_num_batched_tokens,
            kv_capacity=kv_capacity)
        if batch_size != args.batch_size:
            print(f"[Tracer] Requested batch {args.batch_size} collides with the taint registry "
                  f"or exceeds one scheduler step; using {batch_size}")
        if prompt_len < len(prompt_ids):
            print(f"[Tracer] Prompt truncated from {len(prompt_ids)} to {prompt_len} tokens so a "
                  f"collision-free batch fits one scheduler step")
        print(f"[Tracer] Trace batch: {batch_size} prompts x {prompt_len} tokens = "
              f"{batch_size * prompt_len} tokens (registry: {len(TAINT_REGISTRY)} entries)")
        from vllm.inputs import TokensPrompt
        prompts = [TokensPrompt(prompt_token_ids=list(prompt_ids[:prompt_len]))] * batch_size

        # Warm-up: run the exact traced prompts once with taint tracking still
        # off, so one-time work keyed on these shapes (Triton autotuning of
        # the linear-attention kernels, lazy allocations) happens outside the
        # profiled generate instead of inside the first layer's trace.
        #
        # It also shows whether the engine runs the batch in one scheduler
        # step, which the registry check above assumes: a step with only part
        # of the batch runs a request count of its own, which may be a config
        # value (Kimi-K3 on vLLM 0.34: 88 of 94 requests fit its KV cache,
        # then a step of 6, and 6 is MODEL_CONFIG). The capacity estimate
        # cannot see every engine's block accounting (hybrid models' KV-cache
        # groups share one block pool), so the batch is chosen again under the
        # request count the engine actually ran at once.
        from doolyprof.tracer.config import observe_scheduled_requests, observed_scheduled_requests
        for _attempt in range(3):
            observe_scheduled_requests()
            llm.generate(prompts, sampling_params=sampling_params)
            _steps = observed_scheduled_requests()
            _partial = [n for n in _steps if 0 < n < batch_size]  # an empty step runs no request
            if not _partial:
                break
            _at_once = max(_partial)
            print(f"[Tracer] The engine ran the {batch_size}-request batch in steps of "
                  f"{sorted(set(_steps), reverse=True)} requests; choosing a batch of at most {_at_once}")
            _cap = (lambda p, _k=kv_capacity, _n=_at_once: min(_n, _k(p)) if _k else _n)
            kv_capacity = _cap
            batch_size, prompt_len = choose_trace_batch(
                args.batch_size, len(prompt_ids), sched.max_num_seqs, sched.max_num_batched_tokens,
                kv_capacity=kv_capacity)
            print(f"[Tracer] Trace batch: {batch_size} prompts x {prompt_len} tokens = "
                  f"{batch_size * prompt_len} tokens")
            prompts = [TokensPrompt(prompt_token_ids=list(prompt_ids[:prompt_len]))] * batch_size
        else:
            raise RuntimeError(f"the engine did not run a {batch_size}-request trace batch in one step")
        print("[Tracer] Warm-up generate done")

        # Enable taint tracking now - warm-up/compilation is done
        enable_taint_tracking()
        print("\n[Tracer] Taint tracking ENABLED for actual generation")

        # on_trace_ready runs when the profile context exits, which it also does
        # when generate() raises: without this flag a failed run leaves a partial
        # trace that looks like a valid one (and that the profiler would use).
        generate_finished = [False]

        def dooly_trace_handler(dir_name: str, model_name: str, tp: int):
            """torch.profiler handler that writes to a predictable filename
            ``{model_name}_{tp}_dooly_trace.json`` and appends the
            TAINT_REGISTRY snapshot as a top-level key. Mirrors the pattern
            of torch.profiler.tensorboard_trace_handler.

            The handler runs after the profile context exits, at which point
            TAINT_REGISTRY is fully populated by the preceding generate()
            call, so the snapshot is complete.
            """
            import json as _json

            safe_model = model_name.replace("/", "_")
            dp_tag = f"_dp{FAKE_DP}" if FAKE_DP > 1 else ""
            file_name = f"{safe_model}_{tp}{dp_tag}_dooly_trace.json"

            def handler_fn(prof) -> None:
                if not generate_finished[0]:
                    print("[Tracer] generate() did not finish; no trace written", flush=True)
                    return
                if not os.path.isdir(dir_name):
                    try:
                        os.makedirs(dir_name, exist_ok=True)
                    except Exception as e:
                        raise RuntimeError(f"Can't create directory: {dir_name}") from e
                trace_path = os.path.join(dir_name, file_name)
                prof.export_chrome_trace(trace_path)

                # Inject the taint registry as a top-level key.
                with open(trace_path, "r") as f:
                    data = _json.load(f)
                # Keep only the event categories the trace parser consumes
                # (cpu ops, GPU kernels, and the tracer's own annotations).
                # Kineto's async-correlation ("ac2g"), CUDA runtime/driver,
                # overhead and memcpy records are the bulk of the file and are
                # never read by HierarchicalTraceParser.
                _KEEP = {"cpu_op", "kernel", "user_annotation"}
                _n_all = len(data.get("traceEvents", []))
                # Driver-API launches (Triton) carry no External id; link them
                # to their enclosing CPU event while launch records still exist.
                from doolyprof.tracer.kineto_links import attach_launch_external_ids
                _n_linked = attach_launch_external_ids(data.get("traceEvents", []))
                print(f"[Tracer] Linked {_n_linked} kernels without External id to their enclosing CPU event")
                data["traceEvents"] = [e for e in data.get("traceEvents", []) if e.get("cat") in _KEEP]
                print(f"[Tracer] Kept {len(data['traceEvents'])}/{_n_all} trace events ({sorted(_KEEP)})")
                data["dooly_taint_registry"] = {
                    str(v): str(t) for v, t in TAINT_REGISTRY.items()
                    if not isinstance(t, dict)
                }
                with open(trace_path, "w") as f:
                    _json.dump(data, f)
                print(f"[Tracer] Wrote trace + taint registry "
                      f"({len(data['dooly_taint_registry'])} entries) to {trace_path}")

            return handler_fn

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            # with_stack=True,
            record_shapes=True,
            on_trace_ready=dooly_trace_handler(args.trace_dir, args.model, args.tensor_parallel_size),
        ) as prof:
            llm.generate(prompts, sampling_params=sampling_params)
            generate_finished[0] = True

        print(f"[Tracer] Done tracing {args.model}, output saved to {args.trace_dir}")
        from doolyprof.tracer.tensor import qdebug_dump
        qdebug_dump()  # DOOLY_QDEBUG=1 only

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
        # vLLM's own teardown, not torch.distributed's: it also drops vLLM's references to
        # the groups, so an engine whose shutdown runs the same teardown at exit finds
        # nothing left to destroy (instead of destroying groups torch no longer has)
        from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
        destroy_model_parallel()
        destroy_distributed_environment()
