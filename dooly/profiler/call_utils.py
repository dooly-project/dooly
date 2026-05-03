"""Helpers for replaying traced module calls during profiling."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple

from dooly.profiler.module import ModuleInfo


def _forward_signature(callable_obj: Callable) -> Optional[inspect.Signature]:
    """Return the real forward signature for modules when possible."""
    target = getattr(callable_obj, "forward", callable_obj)
    try:
        return inspect.signature(target)
    except (TypeError, ValueError):
        return None


def prepare_module_call_args(
    callable_obj: Callable,
    module: ModuleInfo,
    inputs: List[Any],
    injected_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Any], Dict[str, Any]]:
    """Pack trace-derived module inputs for a real module ``forward`` call.

    MODULE annotations only record tensor inputs. SGLang modules commonly have a
    non-tensor ``forward_batch`` argument in the middle of the signature, so a
    simple ``callable(*inputs)`` shifts later tensor inputs into the wrong slot.
    Align the captured tensors against the forward signature and inject runtime
    objects such as ``forward_batch`` by name.
    """
    injected_kwargs = {
        name: value
        for name, value in (injected_kwargs or {}).items()
        if value is not None
    }

    captured_positional: List[Any] = []
    captured_kwargs: Dict[str, Any] = {}

    for profile_input, value in zip(module.inputs, inputs):
        arg_name = getattr(profile_input, "arg_name", None)
        if arg_name:
            captured_kwargs[arg_name] = value
        else:
            captured_positional.append(value)

    if len(inputs) > len(module.inputs):
        captured_positional.extend(inputs[len(module.inputs):])

    sig = _forward_signature(callable_obj)
    if sig is None:
        captured_kwargs.update(
            {k: v for k, v in injected_kwargs.items() if k not in captured_kwargs}
        )
        return captured_positional, captured_kwargs

    params = list(sig.parameters.values())
    positional_params = [
        p
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                      inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    keywordable = {
        p.name
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                      inspect.Parameter.KEYWORD_ONLY)
    }
    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params
    )
    accepts_var_args = any(
        p.kind == inspect.Parameter.VAR_POSITIONAL for p in params
    )

    assigned: Dict[str, Any] = {}
    passthrough_kwargs: Dict[str, Any] = {}
    for name, value in captured_kwargs.items():
        if name in keywordable:
            assigned[name] = value
        elif accepts_var_kwargs:
            passthrough_kwargs[name] = value

    pos_idx = 0
    for param in positional_params:
        if param.name in assigned:
            continue
        if param.name in injected_kwargs:
            continue
        if pos_idx >= len(captured_positional):
            break
        assigned[param.name] = captured_positional[pos_idx]
        pos_idx += 1

    extra_positional = captured_positional[pos_idx:]

    for name, value in injected_kwargs.items():
        if name in assigned:
            continue
        if name in keywordable:
            assigned[name] = value
        # Deliberately do NOT fall through to accepts_var_kwargs: generic
        # dispatchers like sglang's MultiPlatformOp.forward(*args, **kwargs)
        # forward kwargs into a concrete backend method (e.g. forward_cuda)
        # that doesn't accept forward_batch. Modules that genuinely consume
        # forward_batch (LlamaAttention, RadixAttention) name it explicitly.

    for param in params:
        if param.name in assigned:
            continue
        if param.default is not inspect.Parameter.empty:
            continue
        # First decoder layer calls pass residual=None at runtime, which the
        # tensor-only module annotation omits. Replay that explicit None.
        if param.name == "residual":
            assigned[param.name] = None

    args: List[Any] = []
    kwargs: Dict[str, Any] = {}
    for param in params:
        if param.kind == inspect.Parameter.POSITIONAL_ONLY:
            if param.name in assigned:
                args.append(assigned[param.name])
        elif param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            inspect.Parameter.KEYWORD_ONLY):
            if param.name in assigned:
                kwargs[param.name] = assigned[param.name]

    if accepts_var_args:
        args.extend(extra_positional)
    kwargs.update(passthrough_kwargs)
    return args, kwargs
