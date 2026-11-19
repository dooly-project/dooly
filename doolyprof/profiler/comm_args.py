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


def swept_size_lists(meta: dict, labels) -> list:
    """List arguments made of swept integers (e.g. the per-rank ``sizes`` of
    all_gatherv / reduce_scatterv), as ``(traced values, taint)`` pairs."""
    found = []

    def visit(v):
        if isinstance(v, list) and v and all(isinstance(x, dict) and x.get("taint") in labels for x in v):
            found.append(([x["int"] for x in v], v[0]["taint"]))
        elif isinstance(v, list):
            for x in v:
                visit(x)
        elif isinstance(v, dict) and "int" not in v and "shape" not in v:
            for x in v.values():
                visit(x)

    for v in meta.get("args", {}).values():
        visit(v)
    return found


def swept_payload_dims(meta: dict, shape, dimensions, swept, labels) -> list:
    """Which dims of a payload follow the workload sweep: a dim with a swept
    taint (``dimensions[i] in swept``), or one equal to a per-rank size of a
    swept list argument or their sum. The second case covers payloads the
    tracer saw as plain tensors (an EP dispatch runs inside the MoE custom op)
    whose per-rank sizes kept the token taint."""
    lists = [values for values, _ in swept_size_lists(meta, labels)]
    tied = {v for values in lists for v in values} | {sum(values) for values in lists}
    return [d in swept or size in tied for size, d in zip(shape, dimensions)]


def describe_exception(e) -> str:
    """The exception's message, or (for a bare ``assert``) its type and where it was raised."""
    if str(e):
        return str(e)
    import traceback
    frames = traceback.extract_tb(e.__traceback__)
    where = f" at {frames[-1].filename}:{frames[-1].lineno} ({frames[-1].name})" if frames else ""
    return f"{type(e).__name__}{where}"
