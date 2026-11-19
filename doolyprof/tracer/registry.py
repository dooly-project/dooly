"""Global taint registry for value→taint mapping.

This module maintains a global registry that maps dimension values to their semantic taints.
It is populated from two sources:
1. Config tainting: MODEL_CONFIG values from model configurations
2. Reshape operations: MIX values created during reshape with their component decomposition

The registry enables efficient taint lookup when resolving -1 dimensions in reshape operations.
"""

from typing import Optional, Dict, Set
from .types import Taint

__all__ = ['TAINT_REGISTRY', 'register_taint', 'lookup_taint', 'clear_registry',
           'choose_trace_batch_size', 'choose_trace_batch']

# Global taint registry: value → taint info
# - For pure taints: value → Taint object
# - For MIX taints: value → {'taint': Taint('MIX'), 'components': {Taint, ...}}
TAINT_REGISTRY: Dict[int | float, any] = {}


def register_taint(value: int | float, taint: Taint, components: Optional[Dict[Taint, int]] = None):
    """Register a value→taint mapping in the global registry.

    Args:
        value: The numeric value (dimension size)
        taint: The taint label (NUM_TOK, MODEL_CONFIG, MIX, etc.) or DimTaint object
       components: For MIX taints, the set of component taints
           Example: {Taint('NUM_TOK'), Taint('MODEL_CONFIG')}
           
        # MIX taint with components
        register_taint(240, Taint('MIX'), components={
            Taint('NUM_TOK'),
            Taint('MODEL_CONFIG')
            })
            
    Examples:
        # Pure taint
        register_taint(5120, Taint('MODEL_CONFIG'))

        # MIX taint with components
        register_taint(240, Taint('MIX'), components={
            Taint('NUM_TOK'),
            Taint('MODEL_CONFIG')
        })

        # DimTaint with merge_history (components extracted automatically)
        register_taint(240, DimTaint(Taint('MIX'), merge_history={...}))

    Raises:
        ValueError: If value already exists in registry with different taint/components

    Note:
        This function always registers taints regardless of the global tracking flag.
        The flag only controls arithmetic operations (_binop), not config/reshape registration.
        This ensures config values and reshape dimension semantics are always captured,
        while preventing false taints from warm-up arithmetic operations.
    """
    if value == 1 or value == 0 or value == -1:
        return

    # If taint is a DimTaint object, extract the underlying taint and components
    from .types import DimTaint
    if isinstance(taint, DimTaint):
        if components is None:
            components = taint.merge_history
        taint = taint.taint

        # Check if we have a nested DimTaint
        if isinstance(taint, DimTaint):
            print(f"[ERROR] Nested DimTaint passed to register_taint for value {value}")
            print(f"  Outer DimTaint.taint = {taint}")
            print(f"  Inner DimTaint.taint = {taint.taint}")
            # Extract the actual Taint from the nested structure
            while isinstance(taint, DimTaint):
                taint = taint.taint

    # Check for conflicts if value already exists in registry
    if value in TAINT_REGISTRY:
        existing = TAINT_REGISTRY[value]
        is_new_mix = components is not None and len(components) > 1
        is_existing_mix = isinstance(existing, dict)

        # Build error message components
        if is_existing_mix:
            existing_str = f"MIX {existing.get('components', {})}"
        else:
            existing_str = str(existing)

        if is_new_mix:
            new_str = f"MIX {components}"
        else:
            new_str = str(taint)

        # Check if they are identical (allow identical re-registration)
        is_identical = False
        if is_existing_mix and is_new_mix:
            is_identical = existing.get('components', {}) == components
        elif not is_existing_mix and not is_new_mix:
            is_identical = existing == taint
           
        # small addition to prevent error when testing whether decode has different call path
        # allow overlap for num_req and num_tok since they are flipped from prefill to decode phase change
        # if not is_identical and ((taint == Taint('NUM_REQS') and existing == Taint('NUM_TOKS')) or (taint == Taint('NUM_TOKS') and existing == Taint('NUM_REQS'))):
        #     print(f"[DEBUG] Allowing NUM_REQ/NUM_TOK overlap for value {value} due to prefill/decode phase change")
        #     TAINT_REGISTRY[value] = taint
        #     return
            
        if not is_identical:
            raise ValueError(
                f"\n{'='*80}\n"
                f"TAINT CONFLICT DETECTED\n"
                f"{'='*80}\n"
                f"Value: {value}\n"
                f"Existing: {existing_str}\n"
                f"Attempting to register as: {new_str}\n"
                f"\n"
                f"This means the same dimension value ({value}) is being used for\n"
                f"different semantic purposes in your model configuration.\n"
                f"\n"
                f"SOLUTIONS:\n"
                f"1. Adjust NUM_TOK or NUM_REQS ranges to avoid overlap with MODEL_CONFIG\n"
                f"2. Use different values for different semantic dimensions\n"
                f"3. Check your config values and input buffer sizes\n"
                f"{'='*80}\n"
            )
        # If identical, silently allow re-registration
        return

    # Register the taint (first time)
    if components and len(components) > 1:
        # MIX taint with component history
        TAINT_REGISTRY[value] = {
            'taint': Taint('MIX'),
            'components': components
        }
    else:
        # Pure taint
        TAINT_REGISTRY[value] = taint

    # print(f"[REGISTRY] Registered value {value} with taint {TAINT_REGISTRY[value]}")

def lookup_taint(value: int | float):
    """Look up taint info for a value in the global registry.

    Args:
        value: The numeric value to look up

    Returns:
        - Taint object for pure values
        - Dict with 'taint' and 'components' keys for MIX values
        - None if not found in registry

    Examples:
        >>> register_taint(240, Taint('MIX'), {Taint('NUM_TOK'), Taint('MODEL_CONFIG')})
        >>> lookup_taint(240)
        {'taint': Taint('MIX'), 'components': {Taint('NUM_TOK'), Taint('MODEL_CONFIG')}}
    """
    return TAINT_REGISTRY.get(value)


def clear_registry():
    """Clear the entire registry.

    Useful for testing or when resetting the taint tracking system.
    """
    TAINT_REGISTRY.clear()


def choose_trace_batch_size(requested: int, prompt_len: int, max_num_seqs: int,
                            max_num_batched_tokens: int, registry=None, kv_capacity=None) -> int:
    """Pick how many identical prompts the traced ``generate()`` sends so that
    none of the runtime integers it produces collides with a value already in
    the taint registry.

    By the time the engine is built, the registry holds every integer the
    model config registered as MODEL_CONFIG (including list members such as
    layer-index lists) plus MAX_NUM_REQS / MAX_NUM_TOKS. ``register_taint``
    raises a TAINT CONFLICT when the scheduler then registers one of those
    integers as NUM_REQS or NUM_TOKS, and any raw tensor dimension equal to a
    registered value takes that value's label. One prefill step over ``b``
    prompts of ``prompt_len`` tokens exposes three such integers: ``b``
    (NUM_REQS, seq_lens), ``b + 1`` (query_start_loc / cu_seqlens) and
    ``b * prompt_len`` (NUM_TOKS, positions, slot_mapping). The batch must
    also fit a single scheduler step (``b <= max_num_seqs`` and
    ``b * prompt_len <= max_num_batched_tokens``), otherwise the prefill is
    chunked and NUM_TOKS no longer equals the prompt total. ``kv_capacity``
    (prompt length -> requests the KV cache holds at once, see
    ``kv_cache_capacity``) caps ``b`` likewise: requests the cache cannot hold
    wait for a later step, which then runs a different NUM_REQS.

    Returns the smallest admissible ``b >= requested``, else the largest
    admissible ``b < requested``; raises if no batch size is admissible.
    """
    reg = TAINT_REGISTRY if registry is None else registry
    taken = set(reg)
    upper = min(max_num_seqs, max_num_batched_tokens // max(prompt_len, 1))
    if kv_capacity is not None:
        upper = min(upper, kv_capacity(prompt_len))

    def admissible(b: int) -> bool:
        return 2 <= b <= upper and not ({b, b + 1, b * prompt_len} & taken)

    for b in range(max(requested, 2), upper + 1):
        if admissible(b):
            return b
    for b in range(min(requested, upper), 1, -1):
        if admissible(b):
            return b
    free = next((b for b in range(max(requested, 2), max_num_seqs + 1)
                 if not ({b, b + 1} & taken)), None)
    hint = ""
    if kv_capacity is not None and free is not None and free > upper:
        hint = (f"; the smallest request count clear of the registry is {free}, which needs a larger "
                f"KV cache (raise --gpu-memory-cap)")
    raise ValueError(
        f"No trace batch size in [2, {upper}] avoids the taint registry "
        f"(prompt_len={prompt_len}, one scheduler step and the KV cache hold at most {upper} requests; "
        f"registry values={sorted(v for v in taken if isinstance(v, int))}){hint}")


def choose_trace_batch(requested: int, prompt_len: int, max_num_seqs: int,
                       max_num_batched_tokens: int, registry=None, kv_capacity=None):
    """Like ``choose_trace_batch_size`` but may shorten the prompt.

    Models whose config registers every small integer (Kimi-Linear's layer
    lists cover 2..27, Llama-4 Scout's cover 2..48) leave no admissible batch
    once ``b * prompt_len`` must fit one scheduler step: with the default
    47-token prompt and max_num_batched_tokens=521 the batch cannot exceed 11.
    Prompt content is irrelevant to tracing, only the shapes matter, so the
    prompt is truncated to the longest length that admits a batch.

    Returns ``(batch_size, prompt_len_used)``; raises if no length >= 2 works.
    """
    last_err = None
    for length in range(prompt_len, 1, -1):
        try:
            return choose_trace_batch_size(requested, length, max_num_seqs,
                                           max_num_batched_tokens, registry, kv_capacity), length
        except ValueError as e:
            last_err = e
    raise ValueError(f"No (batch, prompt length) pair avoids the taint registry: {last_err}")


KV_CACHE_LABEL = "KV_CACHE"


def kv_cache_tensors(forward_context) -> list:
    """The KV-cache tensors bound to the engine's layers: every layer in vLLM's
    static forward context (layer name -> layer) holds its cache as
    ``kv_cache``, a tensor or a list / tuple of them (one per virtual engine,
    or a state layer's several state tensors). Empty placeholders are left out."""
    import torch

    found, seen = [], set()

    def visit(obj):
        if isinstance(obj, torch.Tensor):
            if obj.numel() and id(obj) not in seen:
                seen.add(id(obj))
                found.append(obj)
        elif isinstance(obj, (list, tuple)):
            for x in obj:
                visit(x)

    for layer in (forward_context or {}).values():
        visit(getattr(layer, "kv_cache", None))
    return found


def register_kv_cache_dims(tensors, registry=None) -> Dict[int, str]:
    """Label the KV-cache dims no config value explains (the block count, which
    follows the GPU memory left after loading, and a block size the engine
    derives) as KV_CACHE: fixed for the run, never swept, so the profiler
    reads them as fixed dims. Without a label they were ?(396) and ?(768) in
    every cache write of Kimi-K3's KDA and MLA layers. Dims already in the
    registry (head counts and sizes, MODEL_CONFIG) keep their label; registering
    before the trace batch is chosen keeps the batch's integers clear of these
    sizes too. Returns the sizes registered."""
    reg = TAINT_REGISTRY if registry is None else registry
    added = {}
    for t in tensors:
        for size in t.shape:
            size = int(size)
            if size > 1 and size not in reg:
                if registry is None:
                    register_taint(size, Taint(KV_CACHE_LABEL))
                else:
                    reg[size] = Taint(KV_CACHE_LABEL)
                added[size] = KV_CACHE_LABEL
    return added


def _label_name(entry):
    """The label name of a registry entry (Taint, DimTaint or {'taint': ...})."""
    if isinstance(entry, dict):
        entry = entry.get("taint")
    while entry is not None and hasattr(entry, "taint") and not isinstance(entry, Taint):
        entry = entry.taint
    return None if entry is None else str(entry)


def label_module_tensors(model, skip=(), registry=None) -> int:
    """Give every unlabelled dim > 1 of the tensors the model's modules hold
    once loading is done (parameters, buffers, tensor attributes) the
    MODEL_CONFIG label; returns how many tensors got one.

    These tensors exist before any batch does, so their sizes are fixed. Weight
    Parameters are labelled when created (run-tracer's Parameter patch), but
    quantization rebuilds some after loading: vLLM's Marlin FP8 kernels on an
    A100 repack the weights and their scales, and Mixtral-8x7B's fused QKV
    scale [1, 6144] reached marlin_gemm unlabelled, ?(6144) in every layer.
    A size the registry knows by another label (a MAX_NUM_TOKS buffer) is left
    to it, and the ``skip`` tensors (the KV caches, labelled KV_CACHE) are left
    alone. TaintedTensors get their unlabelled dims filled the same way (they
    exist before any batch too): Gemma-3's global rotary cache is built at init
    from a TaintedInt head size and a plain max_position x scaling-factor row
    count, so its 1048576 rows were unlabelled."""
    import torch
    from .types import DimTaint

    reg = TAINT_REGISTRY if registry is None else registry
    skip_ids = {id(t) for t in skip}
    seen, count = set(), 0
    for module in model.modules():
        held = list(module._parameters.values()) + list(module._buffers.values())
        held += [v for k, v in vars(module).items() if isinstance(v, torch.Tensor) and not k.startswith("_")]
        for t in held:
            if (not isinstance(t, torch.Tensor) or id(t) in seen
                    or id(t) in skip_ids or t.dim() == 0):
                continue
            seen.add(id(t))
            sizes = tuple(int(s) for s in torch.Tensor.size(t))
            old = list(getattr(t, "_dim_taints", None) or ())
            if len(old) != len(sizes):  # .data replaced with another shape after the labels were set
                old = [None] * len(sizes)
            new, changed = [], False
            for size, d in zip(sizes, old):
                if (d is None or (isinstance(d, DimTaint) and d.taint is None)) and size > 1 \
                        and _label_name(reg.get(size)) in (None, "MODEL_CONFIG"):
                    d, changed = DimTaint(Taint("MODEL_CONFIG"), merge_history=None), True
                    if reg.get(size) is None:
                        # Registered too, as the Parameter patch does for weight
                        # sizes: a plain copy of the tensor still reads as fixed.
                        # vLLM's rotary embedding replaces its cos_sin_cache with
                        # cos_sin_cache.to(device, dtype) at the first forward;
                        # Gemma-3's global layers (linear RoPE scaling x8) have a
                        # 1048576-row cache no config value gives, ?(1048576).
                        # This runs before the trace batch is chosen, which then
                        # avoids these sizes.
                        if registry is None:
                            register_taint(size, Taint("MODEL_CONFIG"))
                        else:
                            reg[size] = Taint("MODEL_CONFIG")
                new.append(d)
            if changed:
                try:
                    object.__setattr__(t, "_dim_taints", tuple(new))
                    count += 1
                except Exception:  # noqa: BLE001
                    pass
    return count


def kv_cache_capacity(num_blocks, block_size, new_tokens: int = 0):
    """``prompt_len -> requests`` the engine's KV cache holds at once: every
    KV-cache group gives a request whole blocks of ``block_size`` tokens, for
    its prompt and the ``new_tokens`` it generates (at least one block: a
    recurrent-state group takes one per request). None when the engine did not
    report its cache size."""
    if not num_blocks or not block_size:
        return None
    return lambda prompt_len: int(num_blocks) // max(1, -(-(prompt_len + new_tokens) // int(block_size)))
