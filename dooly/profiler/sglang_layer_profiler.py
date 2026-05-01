"""
VLLM Layer Profiler - Profile specific layers/modules using vLLM's full setup.

This profiler:
1. Initializes vLLM's LLM to set up everything (model, caches, metadata builders)
2. Extracts specific layers and their already-bound state caches
3. Uses vLLM's existing metadata builders to create layer-specific metadata
4. Allows profiling full layers OR specific sub-operations (conv1d, SSM, projections)

What we reuse from vLLM:
- Model loading and weight initialization
- State cache allocation and binding
- Layer-specific metadata builders (Mamba2AttentionMetadataBuilder, etc.)
- Forward context setup

What we construct ourselves:
- CommonAttentionMetadata (simple dataclass with batch info)
"""

import os
import pdb

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Literal, Union
from dataclasses import dataclass
from tqdm import tqdm
from contextlib import contextmanager, nullcontext

from dooly.profiler.utils.generate_batch import AttentionBatchConfig

# Apply fake TP patches BEFORE importing SGLang so all modules see patched
# parallel_state at import time. Set DOOLY_FAKE_TP env var before importing
# this module (e.g. in run-profiler.py via --tp flag).
from dooly.profiler.utils.fake_tp_sglang import apply_fake_tp_patches_sglang, get_fake_tp_size

_FAKE_TP_SIZE = get_fake_tp_size()
if _FAKE_TP_SIZE > 1:
    apply_fake_tp_patches_sglang(_FAKE_TP_SIZE)

# from vllm import LLM
# from vllm.config import set_current_vllm_config
# from vllm.forward_context import set_forward_context
# from vllm.v1.attention.backend import CommonAttentionMetadata

from sglang.srt.server_args import ServerArgs, PortArgs
from sglang.srt.managers.scheduler import Scheduler, configure_scheduler
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.mem_cache.common import write_cache_indices


class SgLangLayerProfiler:
    """
    Profile specific layers using vLLM's full infrastructure.

    This class leverages vLLM's LLM to handle all setup:
    - Model loading and weight initialization
    - State cache allocation and binding
    - Metadata builder initialization

    Then allows targeted profiling of specific layers or sub-operations.
    """

    def __init__(
        self,
        model_name: str,
        dtype: str = "bfloat16",
        enforce_eager: bool = True,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 2048,
        attn_backend: str = "flashinfer",
        gpu: str = "0"
    ):
        """
        Initialize the profiler by creating a vLLM LLM instance.

        Args:
            model_name: HuggingFace model name
            dtype: Data type ("float16" or "bfloat16")
            enforce_eager: Disable CUDA graphs for easier profiling
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum sequence length
        """
        self.model_name = model_name
        self.dtype = dtype
        self.torch_dtype = getattr(torch, dtype)

        # Use the requested physical CUDA device instead of remapping it via
        # CUDA_VISIBLE_DEVICES.
        gpu_id = int(gpu)
        torch.cuda.set_device(gpu_id)
        self.device = torch.device(f"cuda:{gpu_id}")

        print(f"[SGLangLayerProfiler] Initializing SgLang with model: {model_name}")

        # Fake TP size (patches applied at module level, see top of file)
        self.tp_size = _FAKE_TP_SIZE

        # If we want to access model_runner, it runs in a separate process instead of sglang.Engine
        # instead just use the Engine knobs to initialize scheduler instead
        server_args = ServerArgs(
            model_path=model_name,
            dtype=dtype,
            load_format="dummy",
            mem_fraction_static=gpu_memory_utilization,
            context_length=max_model_len,
            attention_backend=attn_backend,
            tp_size=1,
            trust_remote_code=True,
            disable_cuda_graph=True,
            disable_piecewise_cuda_graph=True,
        )

        port_args = PortArgs.init_new(server_args)

        configure_scheduler(
            server_args,
            tp_rank=0, attn_cp_rank=0, moe_dp_rank=0,
            moe_ep_rank=0, pp_rank=0, dp_rank=None,
        )

        self.scheduler = Scheduler(
            server_args=server_args,
            port_args=port_args,
            gpu_id=gpu_id,
            tp_rank=0,
            moe_ep_rank=0,
            pp_rank=0,
            attn_cp_rank=0,
            moe_dp_rank=0,
            dp_rank=None,
        )

        self.model_runner = self.scheduler.tp_worker.model_runner
        self.model = self.model_runner.model
        self.model_config = self.model_runner.model_config
        self.server_args = server_args
        self.req_to_token_pool = self.model_runner.req_to_token_pool
        self.attn_backend = self.model_runner.attn_backend

        # Use num_blocks from vLLM's KV cache config
        # vLLM already calculates this based on available GPU memory
        self.actual_num_blocks = self.model_runner.max_total_num_tokens // self.model_runner.page_size 
        self.actual_num_tokens = self.model_runner.max_total_num_tokens

    # Don't need to do _get_actual_num_tokens -- formula used above is how allocator decies max num blocks

    def get_all_layer_names(self) -> List[str]:
        """Return module paths of all attention layers.

        Mirrors the attention-layer collection loop from
        ModelRunner.init_piecewise_cuda_graphs (which only runs when CUDA
        graphs are enabled, so model_runner.attention_layers is unset here).
        """
        from sglang.srt.model_executor.model_runner import resolve_language_model

        inner_model = resolve_language_model(self.model)  # e.g. LlamaModel
        layers = getattr(inner_model, "layers", None)
        if layers is None:
            return []

        attn_modules: set = set()
        for layer in layers:
            attn = None
            if hasattr(layer, "self_attn"):
                sa = layer.self_attn
                attn = getattr(sa, "attn", None) or getattr(sa, "attn_mqa", None)
            elif hasattr(layer, "attn"):
                attn = layer.attn
            elif hasattr(layer, "linear_attn"):
                la = layer.linear_attn
                attn = getattr(la, "attn", la)
            elif hasattr(layer, "attention"):
                ab = layer.attention
                attn = getattr(ab, "attn", ab)
            elif hasattr(layer, "mixer"):
                mixer = layer.mixer
                if hasattr(mixer, "attn"):
                    attn = mixer.attn
                elif hasattr(layer, "_forward_mamba"):
                    attn = layer
            if attn is not None:
                attn_modules.add(attn)

        return [
            name for name, module in self.model.named_modules()
            if module in attn_modules
        ]

    def get_layer_name_by_index(self, layer_idx: int, layer_type: str = "attention") -> str:
        """
        Get the actual layer name from attn_groups based on layer index.

        SgLang stores layer names like "model.layers.0.self_attn" in attn_groups.
        """
        all_names = self.get_all_layer_names()

        # Pattern to match: ".{layer_idx}." in the name
        search_pattern = f".{layer_idx}."

        # First try to find exact match with layer type
        for name in all_names:
            if search_pattern in name:
                if layer_type == "attention" and "attn" in name.lower():
                    return name
                elif layer_type == "mamba" and ("mamba" in name.lower() or "mixer" in name.lower()):
                    return name

        # Fallback: just match by index
        for name in all_names:
            if search_pattern in name:
                return name

        raise ValueError(f"Cannot find layer {layer_idx} (type={layer_type}) in attn_groups. "
                         f"Available: {all_names[:5]}...")

    def get_layer_name_by_module(self, operation_name: str, name_to_module: Dict[str, List[str]]) -> Optional[str]:
        """Find the layer name in attn_groups that matches this module operation.

        Works for ANY module type (attention, mamba, moe, etc.) - uses attn_groups
        as the source of truth for which modules need forward context.
        """
        import re

        module_class = operation_name
        instance_idx = 0
        if '_' in module_class and module_class.rsplit('_', 1)[-1].isdigit():
            instance_idx = int(module_class.rsplit('_', 1)[1])
            module_class = module_class.rsplit('_', 1)[0]

        paths = name_to_module.get(module_class)
        if not paths:
            return None
        # Pick the instance by index; clamp to valid range
        callable_path = paths[min(instance_idx, len(paths) - 1)]

        all_layer_names = self.get_all_layer_names()

        if callable_path in all_layer_names:
            return callable_path

        for name in all_layer_names:
            if callable_path in name or name in callable_path:
                return name
            match = re.search(r'\.(\d+)\.', callable_path)
            if match:
                layer_idx = match.group(0)
                if layer_idx in name:
                    return name

        return None

    def build_fake_forward_batch(
        self,
        batch_config: AttentionBatchConfig,
    ) -> Optional[Tuple[ForwardBatch, torch.Tensor]]:
        """Build a fake ForwardBatch and configure attn_backend for a batch config.

        Equivalent to vLLM's build_layer_metadata — allocates fake KV cache
        slots, populates req_to_token_pool, constructs ForwardBatch, and calls
        init_forward_metadata so the shared attn_backend is ready for any
        attention layer forward call.

        Returns (fake_batch, token_indices) for cleanup, or None if the batch
        config exceeds available capacity.
        """
        batch_size = batch_config.batch_size
        is_prefill = batch_config.is_prefill
        kv_cache_size = batch_config.kv_cache_size
        num_new_tokens = batch_config.prefill_chunk_size if is_prefill else 1
        total_seq_len = kv_cache_size + num_new_tokens
        total_tokens = batch_size * total_seq_len

        allocator = self.model_runner.token_to_kv_pool_allocator
        if allocator.available_size() < total_tokens:
            return None

        # Use fixed row indices 0..batch_size-1 — no req lifecycle needed
        req_pool_indices_cpu = torch.arange(batch_size, dtype=torch.int64)
        req_pool_indices_device = req_pool_indices_cpu.to(self.device, non_blocking=True)

        # Allocate all KV cache token slots (prefix + new tokens)
        all_token_indices = allocator.alloc(total_tokens)
        if all_token_indices is None:
            return None

        # Split flat allocation into per-request prefix (history) and extend (new) parts
        prefix_tensors = []
        extend_locs = []
        for i in range(batch_size):
            chunk = all_token_indices[i * total_seq_len : (i + 1) * total_seq_len]
            prefix_tensors.append(chunk[:kv_cache_size])
            extend_locs.append(chunk[kv_cache_size:])
        out_cache_loc = torch.cat(extend_locs)

        # Build index tensors for write_cache_indices
        prefix_lens_cpu = torch.tensor([kv_cache_size] * batch_size, dtype=torch.int64)
        prefix_lens_device = prefix_lens_cpu.to(self.device, non_blocking=True)
        seq_lens_cpu = torch.tensor([total_seq_len] * batch_size, dtype=torch.int64)
        seq_lens_device = seq_lens_cpu.to(self.device, non_blocking=True)
        extend_lens_cpu = torch.tensor([num_new_tokens] * batch_size, dtype=torch.int64)
        extend_lens_device = extend_lens_cpu.to(self.device, non_blocking=True)

        # Populate req_to_token_pool: maps (req_slot, token_pos) → kv_cache_slot
        write_cache_indices(
            out_cache_loc,
            req_pool_indices_device,
            req_pool_indices_cpu,
            prefix_lens_device,
            prefix_lens_cpu,
            seq_lens_device,
            seq_lens_cpu,
            extend_lens_device,
            extend_lens_cpu,
            prefix_tensors,
            self.req_to_token_pool,
        )

        # Build ForwardBatch — the struct init_forward_metadata reads
        num_query_tokens = batch_size * num_new_tokens
        fake_batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND if is_prefill else ForwardMode.DECODE,
            batch_size=batch_size,
            input_ids=torch.zeros(num_query_tokens, dtype=torch.int64, device=self.device),
            req_pool_indices=req_pool_indices_device.to(torch.int32),
            seq_lens=seq_lens_device.to(torch.int32),
            out_cache_loc=out_cache_loc,
            seq_lens_cpu=seq_lens_cpu.tolist(),
            seq_lens_sum=int(batch_size * total_seq_len),
            extend_prefix_lens=prefix_lens_device.to(torch.int32),
            extend_prefix_lens_cpu=[kv_cache_size] * batch_size,
            extend_seq_lens=extend_lens_device.to(torch.int32),
            extend_seq_lens_cpu=[num_new_tokens] * batch_size,
            encoder_lens=None,
            spec_info=None,
            attn_backend=self.attn_backend,
        )

        # Configure the shared attn_backend — equivalent to vLLM's builder.build()
        self.attn_backend.init_forward_metadata(fake_batch)

        return fake_batch, all_token_indices

    def close(self):
        import gc

        try:
            self.scheduler.shutdown()
        except Exception:
            pass

        try:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except Exception:
            pass

        del self.scheduler

        gc.collect()
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def _free_fake_batch(self, token_indices: torch.Tensor) -> None:
        """Return token KV cache slots to the allocator after a profiling pass."""
        self.model_runner.token_to_kv_pool_allocator.free(token_indices)


def make_dict(model):
    """Map each module class name to an ordered list of all its paths in the model.

    named_modules() visits in depth-first order, which matches the _N suffix
    that the tracer assigns to repeated class instances (e.g. GemmaRMSNorm_0,
    GemmaRMSNorm_1, ...). Storing all paths lets callers pick the correct
    instance by index rather than always returning the first one.
    """
    ret = {}
    for call_path, module in model.named_modules():
        module_name = type(module).__name__
        ret.setdefault(module_name, []).append(call_path)
    return ret

def _run_profiler_in_subprocess(model_name: str, dtype: str, attn_backend: str):
    """Target function for subprocess - all CUDA work happens here."""
    import torch
    print(f"\n=== Run {model_name} ({dtype}) (subprocess PID: {os.getpid()}) ===")

    vlp = SgLangLayerProfiler(
        model_name=model_name,
        dtype=dtype,
        gpu_memory_utilization=0.9,
        attn_backend=attn_backend,
    )

    # Do your profiling work here...

    print("\nClosing SGLang instance...")
    vlp.close()
    print(f"=== Run {model_name} ({dtype}) ({attn_backend}) complete ===\n")

def run_profiler(model_name: str, dtype: str, attn_backend: str):
    # All CUDA work happens in subprocess
    p = mp.Process(target=_run_profiler_in_subprocess, args=(model_name, dtype, attn_backend))
    p.start()
    p.join()  # Wait for subprocess to finish - GPU memory fully freed on exit



if __name__ == "__main__":
    
    test_vlp = SgLangLayerProfiler(
        model_name="facebook/opt-125m",
        dtype="bfloat16",
        gpu_memory_utilization=0.2,
        attn_backend="flashinfer",
    )

    print(f"Actual num blocks: {test_vlp.actual_num_blocks}")
