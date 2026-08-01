"""Small compatibility helpers for SGLang model configs."""

from __future__ import annotations

from typing import Any


def normalize_nested_rope_parameters_for_sglang(model_config: Any) -> bool:
    """Normalize nested HF RoPE config dictionaries for this SGLang build.

    Some HF configs represent per-attention-type RoPE as nested dictionaries,
    e.g. {"full_attention": {"rope_type": "default", ...}}. SGLang model
    code may expect optional no-op fields such as ``factor`` to exist when it
    converts one of those nested default-RoPE blocks into its internal format.

    This is schema-based: it does not check model names or classes. It only
    fills ``factor=1.0`` for nested attention RoPE blocks that are explicitly
    default/no-scaling and omit the field.
    """

    visited: set[int] = set()

    def patch_rope_block(block: Any) -> bool:
        if not isinstance(block, dict):
            return False
        rope_type = dict.get(block, "rope_type", dict.get(block, "type", "default"))
        if rope_type not in (None, "default"):
            return False
        if "factor" in block:
            return False
        block["factor"] = 1.0
        return True

    def patch_mapping(mapping: Any) -> bool:
        if not isinstance(mapping, dict):
            return False

        changed = False
        for key, value in list(dict.items(mapping)):
            if isinstance(key, str) and key.endswith("_attention"):
                changed = patch_rope_block(value) or changed

        # Tainted config dictionaries cache nested mappings after access. If the
        # cache was already populated, patch that cached copy too.
        cache = getattr(mapping, "_taint_cache", None)
        if isinstance(cache, dict):
            for key, value in list(cache.items()):
                if isinstance(key, str) and key.endswith("_attention"):
                    changed = patch_rope_block(value) or changed
            if changed:
                cache.pop("factor", None)

        return changed

    def visit(obj: Any) -> bool:
        if obj is None:
            return False
        obj_id = id(obj)
        if obj_id in visited:
            return False
        visited.add(obj_id)

        changed = False

        if isinstance(obj, dict):
            changed = patch_mapping(obj)
            for value in list(dict.values(obj)):
                if isinstance(value, dict):
                    changed = visit(value) or changed
            return changed

        try:
            raw_attrs = vars(obj)
        except TypeError:
            raw_attrs = {}

        changed = patch_mapping(raw_attrs.get("rope_parameters")) or changed

        cache = raw_attrs.get("_taint_cache")
        if isinstance(cache, dict):
            changed = patch_mapping(cache.get("rope_parameters")) or changed
            if changed:
                cache.pop("rope_parameters", None)

        for attr_name in (
            "hf_config",
            "hf_text_config",
            "text_config",
            "thinker_config",
        ):
            if attr_name in raw_attrs:
                changed = visit(raw_attrs[attr_name]) or changed

        return changed

    return visit(model_config)


def patch_sglang_model_loader_rope_parameters() -> bool:
    """Patch SGLang model initialization to normalize nested RoPE configs."""

    try:
        import sglang.srt.model_loader.loader as sglang_loader
    except ImportError:
        return False

    original_initialize_model = getattr(sglang_loader, "_initialize_model", None)
    if original_initialize_model is None:
        return False
    if getattr(original_initialize_model, "_dooly_rope_normalized", False):
        return False

    def patched_initialize_model(model_config, *args, **kwargs):
        normalize_nested_rope_parameters_for_sglang(model_config)
        return original_initialize_model(model_config, *args, **kwargs)

    patched_initialize_model._dooly_rope_normalized = True
    sglang_loader._initialize_model = patched_initialize_model
    return True


__all__ = [
    "normalize_nested_rope_parameters_for_sglang",
    "patch_sglang_model_loader_rope_parameters",
]
