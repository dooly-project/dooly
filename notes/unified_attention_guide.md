# Unified Attention Usage Guide

This guide explains how to use the [run_unified_attn.py](file:///home/jkim38/my-eval-lib/run_unified_attn.py) script to test and profile different attention backends (`TRITON_ATTN`, `FLASH_ATTN`, `FLASHINFER`) using the `torch.ops.vllm.unified_attention_with_output` operator.

## Prerequisites
- Working vLLM installation
- [unified_attention_guide.md](file:///home/jkim38/.gemini/antigravity/brain/f09f6e5f-6033-4de2-8d38-f9ab1a628fee/unified_attention_guide.md) guidelines

## Key Concepts with Unified Attention
The [unified_attention](file:///home/jkim38/miniforge3/envs/utns_sim/lib/python3.11/site-packages/vllm/attention/layer.py#793-804) interface requires specific preparation steps that are usually handled by the `ModelRunner`. Since we are calling the operator directly, we must manually:
1.  **Set Global Context**: Use `vllm.config.set_current_vllm_config` to establish the global configuration context.
2.  **Initialize Layer**: Create the [Attention](file:///home/jkim38/miniforge3/envs/utns_sim/lib/python3.11/site-packages/vllm/attention/layer.py#143-497) layer with a mock `ModelConfig`.
3.  **Setup KV Cache**: Manually allocate KV cache tensors with the correct backend-specific shape.
4.  **Build Metadata**: Construct [AttentionMetadata](file:///home/jkim38/miniforge3/envs/utns_sim/lib/python3.11/site-packages/vllm/v1/attention/backend.py#279-281) using [AttentionMetadataBuilder](file:///home/jkim38/miniforge3/envs/utns_sim/lib/python3.11/site-packages/vllm/v1/attention/backend.py#433-576).
5.  **Inject Context**: Use [set_forward_context](file:///home/jkim38/miniforge3/envs/utns_sim/lib/python3.11/site-packages/vllm/forward_context.py#312-421) to pass the metadata to the operator.

## Script Usage

### Running the Script
The script supports command-line arguments to select the backend and data type:

```bash
# Default (Triton Attention, float16)
python run_unified_attn.py

# Explicit Backend Selection
python run_unified_attn.py --backend FLASH_ATTN --dtype float16
python run_unified_attn.py --backend TRITON_ATTN --dtype float16
python run_unified_attn.py --backend FLASHINFER --dtype bfloat16
```

> [!IMPORTANT]
> **Dtype Requirement**: The [Attention](file:///home/jkim38/miniforge3/envs/utns_sim/lib/python3.11/site-packages/vllm/attention/layer.py#143-497) class uses `torch.get_default_dtype()` during initialization to determine the backend precision validation. You **must** call `torch.set_default_dtype(your_dtype)` before instantiating the layer if you are using non-float32 types (like float16 or bfloat16), otherwise backend selection might fail.

### Automated KV Cache Shape
The script automatically determines the correct KV cache shape for the selected backend:
```python
# run_unified_attn.py
kv_cache_shape = layer.attn_backend.get_kv_cache_shape(
    num_blocks=num_blocks, 
    block_size=16, 
    num_kv_heads=4, 
    head_size=64
)
```

## FAQ

**Does `SchedulerConfig` affect profiling results?** 
In this script, the impact is **minimal**. 
*   We manually construct [AttentionMetadata](file:///home/jkim38/miniforge3/envs/utns_sim/lib/python3.11/site-packages/vllm/v1/attention/backend.py#279-281) (batch size, sequence lengths), effectively bypassing the scheduler's dynamic batching logic.
*   The `SchedulerConfig` is primarily used by the [AttentionMetadataBuilder](file:///home/jkim38/miniforge3/envs/utns_sim/lib/python3.11/site-packages/vllm/v1/attention/backend.py#433-576) during initialization to determine capabilities (e.g., `is_encoder_decoder`, `num_speculative_tokens` for reordering thresholds).
*   However, incorrect values (like missing `max_num_batched_tokens`) can cause validation errors during builder initialization, so a valid mock config is still required.
