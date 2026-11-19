"""Profiling scopes for Triton kernels launched from Python.

A kernel launched straight from a module's forward (``kernel[grid](...)``,
no torch op around it) has no CPU op to anchor to, so the parser anchored it to
the module, and everything else that module runs folded into one unit: one
attention-residual launch at the end of Kimi-K3's model forward made the whole
model a single unit that cannot run on its own.

Each such launch with tainted tensor arguments is wrapped in a
``TRITON: <module>:<function> IN:[...] DTYPES:[...] STATE:<hex> SPEC:<base64 json>``
scope. The parser treats it as a self-tainted anchor, and the profiler launches
the kernel again from SPEC (``doolyprof.profiler.utils.triton_replay``).

A launch gets a scope only when that replay is faithful with tensors the
profiler generates from shapes alone:
  * every tensor argument is floating point: integer and boolean tensors hold
    indices or offsets (sequence boundaries, state-cache slots) whose values
    come from the engine's metadata, and random values would read out of
    bounds or time a different batch;
  * in every tensor the batch dims (NUM_TOKS / NUM_REQS / MIX of them) come
    before all other dims, and the tensor is contiguous, so no stride passed as
    a plain integer depends on the batch;
  * the grid is a tuple (a grid function cannot be replayed), and every
    batch-dependent integer (grid entry or argument) is the size of a tensor
    argument's dim with the same taint, so it can be read off the generated
    tensors;
  * no untainted integer argument equals a batch dim of the launch (it could be
    a batch size that lost its taint).
Anything else keeps the old behaviour: no scope, and the kernel anchors to the
module that launched it.
"""
import base64
import hashlib
import json

import torch

from .types import DimTaint, Taint, TaintedFloat, TaintedInt, split_offset_label

BATCH_LABELS = ("NUM_TOKS", "NUM_REQS", "MIX")


class _NotReplayable(Exception):
    pass


def _label(taint):
    while isinstance(taint, DimTaint):
        taint = taint.taint
    if not isinstance(taint, Taint):
        return None
    parts = split_offset_label(taint)
    if parts is not None and parts[1] != 0:
        # a batch size plus a constant: SPEC can only point at a tensor dim
        # (or hold a fixed value), so such a launch is not replayed
        raise _NotReplayable
    return str(taint)


def _dim_labels(t):
    dts = getattr(t, "_dim_taints", None) or ()
    n = len(torch.Tensor.size(t))
    return [(_label(dts[i]) if i < len(dts) and dts[i] is not None else None) for i in range(n)]


def _shape_str(t):
    from .tensor import TaintedTensor, _format_dim
    if isinstance(t, TaintedTensor):
        return t.taint_str_with_history()
    dts = getattr(t, "_dim_taints", None)
    return "[" + ", ".join(_format_dim(s, dts[i] if dts is not None and i < len(dts) else None)
                           for i, s in enumerate(torch.Tensor.size(t))) + "]"


def launch_annotation(jit_fn, args, kwargs):
    """The TRITON scope name for this launch, or None when it is not replayable
    (or not traced: no tainted tensor argument, taint tracking off)."""
    from . import is_taint_tracking_enabled
    from .tensor import TaintedTensor, in_blackbox
    if not is_taint_tracking_enabled() or in_blackbox():
        # a launch inside a custom op belongs to that op's profiling unit
        return None
    fn = getattr(jit_fn, "fn", None)
    if fn is None or not getattr(fn, "__module__", None):
        return None
    kw = {k: v for k, v in kwargs.items() if k not in ("grid", "warmup")}
    tensors = [a for a in list(args) + list(kw.values()) if isinstance(a, torch.Tensor)]
    if not any(isinstance(t, TaintedTensor) for t in tensors):
        return None
    try:
        with torch._C.DisableTorchFunction():
            return _annotation(fn, args, kw, kwargs.get("grid"), tensors)
    except _NotReplayable:
        return None


def _annotation(fn, args, kw, grid, tensors):
    if grid is None or callable(grid):
        raise _NotReplayable
    batch_dims = {}   # (label, size) -> [tensor index, dim]
    for ti, t in enumerate(tensors):
        if not (t.dtype.is_floating_point and torch.Tensor.is_contiguous(t)):
            raise _NotReplayable
        labels = _dim_labels(t)
        sizes = [int(s) for s in torch.Tensor.size(t)]
        seen_fixed = False
        for d, (lab, size) in enumerate(zip(labels, sizes)):
            if lab in BATCH_LABELS:
                if seen_fixed:
                    raise _NotReplayable
                batch_dims.setdefault((lab, size), [ti, d])
            else:
                seen_fixed = True
    batch_sizes = {size for _, size in batch_dims}

    def scalar(v):
        if isinstance(v, torch.Tensor):
            return {"t": next(i for i, t in enumerate(tensors) if t is v)}
        if isinstance(v, (TaintedInt, TaintedFloat)):
            lab = _label(v.taint)
            if lab in BATCH_LABELS:
                ref = batch_dims.get((lab, int(v)))
                if ref is None:
                    raise _NotReplayable
                return {"dim": ref}
            return {"v": v.value}
        if v is None or isinstance(v, (bool, float, str)):
            return {"v": v}
        if isinstance(v, int):
            if v in batch_sizes:
                raise _NotReplayable
            return {"v": v}
        raise _NotReplayable

    spec = {
        "fn": f"{fn.__module__}:{fn.__name__}",
        "args": [scalar(a) for a in args],
        "kwargs": {k: scalar(v) for k, v in kw.items()},
        "grid": [scalar(g) for g in grid],
    }
    fixed = json.dumps([spec["fn"], [a for a in spec["args"] if "v" in a],
                        {k: v for k, v in spec["kwargs"].items() if "v" in v},
                        [g for g in spec["grid"] if "v" in g]], sort_keys=True, default=str)
    state = hashlib.blake2b(fixed.encode(), digest_size=6).hexdigest()
    return (f"TRITON: {spec['fn']} IN:[{' | '.join(_shape_str(t) for t in tensors)}] "
            f"DTYPES:[{', '.join(str(t.dtype) for t in tensors)}] STATE:{state} "
            f"SPEC:{encode_spec(spec)}")


def encode_spec(spec):
    """SPEC as URL-safe base64 of its JSON: the saved trace turns the double
    quotes of an annotation name into single quotes, which JSON cannot read."""
    return base64.urlsafe_b64encode(json.dumps(spec, default=str).encode()).decode()


def decode_spec(text):
    text = text.strip()
    if text.startswith("{"):
        # written as plain JSON before it was encoded (its double quotes came
        # back single); its strings are import paths, with no quotes of their own
        return json.loads(text.replace("'", '"'))
    return json.loads(base64.urlsafe_b64decode(text.encode()).decode())
