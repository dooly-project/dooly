# vLLM Stateful Layer Workflow: AttentionGroup and Forward Context

This document explains how vLLM manages stateful layers (Attention, Mamba, etc.) through the `AttentionGroup` system and why `forward_context` is required for these layers.

## Overview

vLLM has a unified system for managing layers that maintain persistent state across forward passes. Despite the naming (`AttentionLayerBase`, `attn_groups`), this system handles **all stateful layers**, not just attention:

- **Attention layers**: Maintain KV cache
- **Mamba layers**: Maintain SSM state (conv state, ssm state)
- **Future stateful layers**: Any layer with persistent state

## Class Hierarchy

```
AttentionLayerBase (ABC)
├── Attention
│   ├── LlamaAttention
│   ├── MistralAttention
│   └── ...
└── MambaBase
    ├── MambaMixer   (Mamba1)
    └── MambaMixer2  (Mamba2)
```

### Key Files

| File | Purpose |
|------|---------|
| `vllm/model_executor/layers/attention_layer_base.py` | Abstract base class for all stateful layers |
| `vllm/model_executor/layers/mamba/abstract.py` | `MambaBase` extending `AttentionLayerBase` |
| `vllm/model_executor/layers/mamba/mamba_mixer.py` | `MambaMixer` implementation |
| `vllm/v1/worker/utils.py` | `AttentionGroup` dataclass |
| `vllm/v1/worker/gpu_model_runner.py` | Creates and manages `attn_groups` |

---

## Step 1: Layer Registration via `get_kv_cache_spec()`

Each stateful layer implements `get_kv_cache_spec()` to describe its cache/state requirements:

### Mamba Implementation

```python
# vllm/model_executor/layers/mamba/abstract.py

class MambaBase(AttentionLayerBase):
    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        return MambaSpec(
            shapes=self.get_state_shape(),      # Conv state + SSM state shapes
            dtypes=self.get_state_dtype(),      # Data types for states
            block_size=mamba_block_size,
            page_size_padded=page_size_padded,
            mamba_type=self.mamba_type,         # "mamba1" or "mamba2"
            mamba_cache_mode=...,
        )
```

### Attention Implementation

```python
# vllm/model_executor/layers/attention.py (conceptually)

class Attention(AttentionLayerBase):
    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        return FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            dtype=self.dtype,
        )
```

---

## Step 2: Collecting All Stateful Layers

During model initialization, vLLM iterates through all modules inheriting from `AttentionLayerBase`:

```python
# vllm/v1/worker/gpu/attn_utils.py

def get_kv_cache_spec(vllm_config: VllmConfig) -> dict[str, KVCacheSpec]:
    kv_cache_spec: dict[str, KVCacheSpec] = {}

    # Get ALL layers inheriting from AttentionLayerBase
    # This includes BOTH Attention AND Mamba layers!
    layer_type = AttentionLayerBase
    attn_layers = get_layers_from_vllm_config(vllm_config, layer_type)

    for layer_name, attn_module in attn_layers.items():
        # Call each layer's get_kv_cache_spec()
        if spec := attn_module.get_kv_cache_spec(vllm_config):
            kv_cache_spec[layer_name] = spec

    return kv_cache_spec
    # Example output:
    # {
    #     "model.layers.0.self_attn": FullAttentionSpec(...),
    #     "model.layers.0.mixer": MambaSpec(...),  # Mamba layer!
    #     "model.layers.1.self_attn": FullAttentionSpec(...),
    #     ...
    # }
```

---

## Step 3: Grouping into KV Cache Groups

Layers with compatible specs are grouped together:

```python
# vllm/v1/kv_cache_interface.py

@dataclass
class KVCacheGroupSpec:
    kv_cache_spec: KVCacheSpec    # MambaSpec or AttentionSpec
    layer_names: tuple[str, ...]  # Layers sharing this spec
```

Example for a hybrid model (Jamba/Bamba):
```
KVCacheGroup 0: Attention layers
  - spec: FullAttentionSpec(num_kv_heads=8, head_size=128, ...)
  - layers: ["model.layers.0.self_attn", "model.layers.2.self_attn", ...]

KVCacheGroup 1: Mamba layers
  - spec: MambaSpec(mamba_type="mamba1", ...)
  - layers: ["model.layers.1.mixer", "model.layers.3.mixer", ...]
```

---

## Step 4: Creating AttentionGroups

Each KV cache group gets an `AttentionGroup` with a metadata builder:

```python
# vllm/v1/worker/utils.py

@dataclass
class AttentionGroup:
    backend: type[AttentionBackend]           # FlashAttention, MambaBackend, etc.
    layer_names: list[str]                     # Layers in this group
    kv_cache_spec: KVCacheSpec                 # MambaSpec or AttentionSpec
    kv_cache_group_id: int
    metadata_builders: list[AttentionMetadataBuilder]  # Builds runtime metadata
```

```python
# vllm/v1/worker/gpu_model_runner.py

def create_attn_groups(attn_backends_map, kv_cache_group_id):
    attn_groups = []
    for (attn_backend, kv_cache_spec), layer_names in attn_backends_map.items():
        attn_group = AttentionGroup(
            attn_backend,       # e.g., MambaBackend
            layer_names,        # e.g., ["backbone.layers.0.mixer", ...]
            kv_cache_spec,      # e.g., MambaSpec(...)
            kv_cache_group_id,
        )
        attn_groups.append(attn_group)
    return attn_groups
```

---

## Step 5: Metadata Builder Creates Runtime Context

Each `AttentionGroup` has a metadata builder that converts batch configuration into layer-specific metadata:

```python
# Metadata builder is created from the backend
metadata_builder = attn_backend.get_builder_cls()(
    kv_cache_spec,      # MambaSpec or AttentionSpec
    layer_names,        # Which layers use this metadata
    vllm_config,
    device,
)

# At runtime, builds metadata from CommonAttentionMetadata
layer_metadata = metadata_builder.build(
    common_prefix_len=0,
    common_attn_metadata=CommonAttentionMetadata(
        query_start_loc=...,
        seq_lens=...,
        block_table_tensor=...,
        slot_mapping=...,
        # ... batch/sequence configuration
    ),
)
```

---

## Step 6: Forward Context at Runtime

Before calling a stateful layer's forward(), the context must be set:

```python
# vllm/forward_context.py

@contextmanager
def set_forward_context(
    attn_metadata: dict[str, AttentionMetadata],
    vllm_config: VllmConfig,
    virtual_engine: int = 0,
):
    global _forward_context
    _forward_context = ForwardContext(
        attn_metadata=attn_metadata,
        virtual_engine=virtual_engine,
        no_compile_layers=...,
    )
    try:
        yield
    finally:
        _forward_context = None
```

### MambaMixer's Forward Method

```python
# vllm/model_executor/layers/mamba/mamba_mixer.py

class MambaMixer(MambaBase, CustomOp):
    def forward(self, hidden_states: torch.Tensor, output: torch.Tensor):
        torch.ops.vllm.mamba_mixer(
            output,
            hidden_states,
            self.in_proj.weight,
            # ... other weights
        )

# The actual kernel implementation
@torch.library.custom_op("vllm::mamba_mixer", mutates_args=["output"])
def mamba_mixer(output: torch.Tensor, ...):
    # THIS IS WHERE THE ERROR OCCURS IF CONTEXT NOT SET!
    forward_context: ForwardContext = get_forward_context()  # Line 523

    # Access layer-specific metadata
    attn_metadata = forward_context.attn_metadata

    # Use metadata for cache management
    mamba_cache = ...
```

---

## Why Forward Context is Required

The forward context provides:

1. **Cache/State Location**: Where to read/write persistent state
2. **Batch Configuration**: Which sequences, their lengths, prefill vs decode
3. **Block Tables**: Memory layout for paged state management
4. **Slot Mapping**: Where new state should be written

Without this context, the layer doesn't know:
- Which cache blocks to access
- How many sequences are in the batch
- Whether to run prefill or decode logic

---

## Profiling Implications

When profiling stateful layers, we must:

1. **Detect if layer is in `attn_groups`**: Check if `_find_layer_name()` returns a match
2. **Use `AttentionBatchConfig`**: Provides batch_size, prefill_chunk_size, kv_cache_size
3. **Build layer metadata**: `build_layer_metadata(layer_name, batch_config)`
4. **Set forward context**: Wrap the forward call with `set_forward_context()`

```python
# profiler.py pattern
layer_name = self._find_layer_name(module)  # Check if in attn_groups

if layer_name:
    # Module needs forward context
    layer_metadata = build_layer_metadata(layer_name, batch_config)
    forward_context = {layer_name: layer_metadata}

    with set_forward_context(forward_context, vllm_config):
        module(*inputs)  # Now the layer can access its metadata
else:
    # Simple module, no context needed
    module(*inputs)
```

---

## Summary

| Step | Component | Purpose |
|------|-----------|---------|
| 1 | `get_kv_cache_spec()` | Layer declares its cache requirements |
| 2 | `get_layers_from_vllm_config()` | Collect all `AttentionLayerBase` layers |
| 3 | `KVCacheGroupSpec` | Group compatible layers |
| 4 | `AttentionGroup` | Associate layers with backend + builder |
| 5 | `metadata_builder.build()` | Create runtime metadata from batch config |
| 6 | `set_forward_context()` | Make metadata available to layers |
| 7 | `get_forward_context()` | Layer retrieves its metadata |

The key insight: **`attn_groups` is the single source of truth** for which layers need forward context. Any layer registered there (Attention, Mamba, or future stateful layers) requires context to be set before its forward pass.
