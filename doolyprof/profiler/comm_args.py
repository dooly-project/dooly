"""
Reading the ARGS section of a ``COMM:`` trace annotation.

Traces written by ``doolyprof.profiler.utils.comm_emulation`` carry a Python
literal (single-quoted, so it survives torch's chrome-trace export): the
communication call's bound arguments plus ``api``, ``group``, ``world_size``
and the payload ``dtypes`` (see that module for the encoding). Older traces
carry ``"name=value, ..."`` / ``"-1"`` text; those are read into the same
shape so downstream code has one format.

No vLLM import: the comm profiler worker uses this without fake TP.
"""

import ast
import json


def parse_comm_args(raw) -> dict:
    """ARGS text -> {"api", "group", "world_size", "args": {name: value}, "dtypes"}."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {"api": None, "args": {}, "dtypes": []}
    text = raw.strip()
    if text.startswith("{"):
        for load in (ast.literal_eval, json.loads):
            try:
                meta = load(text)
            except (ValueError, SyntaxError):
                continue
            if isinstance(meta, dict):
                meta.setdefault("args", {})
                meta.setdefault("dtypes", [])
                return meta
    # Legacy "a, name=b, ..." (fake_tp before JSON args): plain ints only.
    args = {}
    for i, part in enumerate(p.strip() for p in text.split(",") if p.strip()):
        key, _, value = part.rpartition("=")
        try:
            args[key or f"arg{i}"] = int(value)
        except ValueError:
            continue
    return {"api": None, "args": args, "dtypes": []}


def identity_int(meta: dict):
    """The collective's integer discriminator stored as ``dim_param``: the first
    untainted integer argument in call order (e.g. the gather/scatter ``dim``),
    None if there is none (e.g. all_reduce). Tainted integers are workload, not
    identity, and are swept instead."""
    for value in meta.get("args", {}).values():
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
    return None


def is_tensor_ref(value) -> bool:
    return isinstance(value, dict) and set(value) == {"tensor"}
