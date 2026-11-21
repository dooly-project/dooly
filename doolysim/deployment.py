"""Which GPU, engine and topology of a profile database a DoolySim run uses.

A DoolyProf database can hold measurements of several GPUs and engine versions
(doolyprof/profiler/db.py): every ``results`` and ``profiled_models`` row carries a ``device``
and an ``engine``, every ``coll_op_results`` row an ``engine`` next to its topology. DoolySim
trains on and simulates one device and engine: the ones given (``--device``, ``--engine``),
else the only ones the database has for the deployment. Collectives use the topology given
(``--topology``), else the only one the deployment's collectives were recorded under.
Databases written before devices were recorded have no such columns; all their rows are used,
as before.
"""
import sqlite3


def columns(conn, table):
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _deployment(model, backend, tp, dtype, kv_cache_dtype, alias=""):
    p = f"{alias}." if alias else ""
    where, params = [], []
    for col, val in (("model_name", model), ("backend", backend), ("tp_degree", tp), ("dtype", dtype),
                     ("kv_cache_dtype", kv_cache_dtype)):
        if val is not None:
            where.append(f"{p}{col} = ?")
            params.append(val)
    return where, params


def resolve_device_engine(conn, model=None, backend=None, tp=None, dtype=None, kv_cache_dtype=None,
                          device=None, engine=None):
    """(device, engine, labelled): the device and engine to use for this deployment, and whether
    the database records them at all (False for older databases, whose rows are all used).
    Raises ValueError when the choice is missing or ambiguous."""
    if "device" not in columns(conn, "profiled_models"):
        if device or engine:
            raise ValueError("this database records no GPU or engine (it predates them); drop --device / --engine")
        return None, None, False
    where, params = _deployment(model, backend, tp, dtype, kv_cache_dtype)
    sql = "SELECT DISTINCT device, engine FROM profiled_models" + (" WHERE " + " AND ".join(where) if where else "")
    found = sorted(conn.execute(sql, params).fetchall(), key=lambda r: (str(r[0]), str(r[1])))
    picked = [(d, e) for d, e in found if (device is None or d == device) and (engine is None or e == engine)]
    shown = ", ".join(f"{d or '(no device)'} / {e or '(no engine)'}" for d, e in found) or "none"
    if len(picked) == 1:
        return picked[0][0], picked[0][1], True
    if not picked:
        raise ValueError(f"no profile of this deployment for device={device}, engine={engine}; the database has: {shown}")
    raise ValueError(f"the database has this deployment on several GPUs or engines ({shown}); "
                     f"choose one with --device and --engine")


def filter_sql(alias, device, engine, labelled, engine_only=False):
    """(SQL, params) restricting table ``alias`` to the device and engine ('' for older databases)."""
    if not labelled:
        return "", []
    if engine_only:
        return f" AND {alias}.engine IS ?", [engine]
    return f" AND {alias}.device IS ? AND {alias}.engine IS ?", [device, engine]


def resolve_topology(conn, model=None, backend=None, tp=None, dtype=None, kv_cache_dtype=None,
                     device=None, engine=None, labelled=False, topology=None):
    """The topology label to read this deployment's collective timings under: ``topology`` when
    given, else the only label its collectives were recorded under (None without collectives).
    Raises ValueError when they were recorded under several."""
    if topology:
        return topology
    if not columns(conn, "coll_ops") or not columns(conn, "model_collectives"):
        return None
    where, params = _deployment(model, backend, tp, dtype, kv_cache_dtype, alias="pm")
    extra, extra_params = filter_sql("pm", device, engine, labelled)
    sql = ("SELECT DISTINCT co.topology FROM model_collectives mc "
           "JOIN coll_ops co ON co.collective_id = mc.collective_id "
           "JOIN profiled_models pm ON pm.model_id = mc.model_id WHERE 1=1"
           + "".join(f" AND {w}" for w in where) + extra)
    found = sorted(r[0] for r in conn.execute(sql, params + extra_params))
    if len(found) > 1:
        raise ValueError(f"this deployment's collectives were recorded under several topologies ({', '.join(found)}); "
                         f"choose one with --topology")
    return found[0] if found else None
