"""``dooly export`` and ``dooly import``: a profile database as a tree of CSV files, and back.

    profiles/
    ├── compute/<device>/<engine>/<backend>/<model>/<deployment>/
    │   ├── deployment.json      the deployment, and the runs that measured it (versions, options)
    │   ├── model.csv            its operations: file, name, signature, calls per forward pass
    │   ├── collectives.csv      its collectives: topology, group size, calls, how sizes scale
    │   └── <operation>-<signature prefix>.csv   latency of one operation per batch shape
    └── communication/<topology>/<engine>/
        ├── runs.json            the runs that measured them
        └── <collective>.csv     latency per group size and message size

``<device>`` is the GPU model (e.g. ``nvidia-rtx-pro-6000-blackwell-server-edition``),
``<engine>`` the serving engine and version (``vllm-0.33.0``), ``<backend>`` the attention
backend and the version of what implements it (``FLASH_ATTN-fa2``), ``<model>`` the Hugging
Face name (``Qwen/Qwen3.8-27B``, two folders), ``<deployment>``
``tp<N>[-ep<M>]-<dtype>-kv<KV-cache dtype>`` (``tp2-bfloat16-kvauto``). Every folder is
self-contained: an operation several models share is written into each model's folder.

An operation is measured once per batch shape on a device and engine. The tensor-parallel
degree of the run that measured it is recorded (``num_tensor_parallel_workers``) but does not
make a second measurement: a deployment of another degree that runs the same operation (the
same signature, so the same shapes and kernels, e.g. RMSNorm over the full hidden size) does the
same work. A database holding two measurements of one batch shape (two runs that did not share a
database) exports the first one it got.

Export only reads the database. Import is the reverse, a union into a database: rows the
database already has (the same signature, batch shape, device and engine) are kept as they
are. Rows written before devices and engines were recorded (doolyprof.profiler.db) are not
exported. Exporting a database, importing the tree into an empty one and exporting that gives
the same files.

``verify_tree()`` (``dooly verify``) checks a tree the same way: it imports it into an empty
database and requires the export to give back exactly its files.

Only the standard library is used.
"""
import csv
import json
import os
import re
import sqlite3

from doolyprof.profiler import db as profile_db

# A measured batch shape (with the signature, device and engine, the key of a latency row), then the
# TP degree of the run that measured it (recorded, not part of the key) and the statistics, in the
# order of the columns of an operation's CSV.
BATCH = ["num_tokens", "num_requests", "is_context", "is_prefill", "kv_cache_size", "prefill_chunk_size",
         "activated_experts", "local_pairs"]
SHAPE = BATCH + ["num_tensor_parallel_workers"]
STATS = ["mean_latency_ms", "median_latency_ms", "min_latency_ms", "max_latency_ms", "std_latency_ms"]
RESULT_COLUMNS = SHAPE + STATS + ["input_shapes", "created_at", "run"]
MODEL_COLUMNS = ["file", "operation_name", "signature_hash", "count", "signature_json"]
COLLECTIVE_COLUMNS = ["signature_hash", "operation_name", "topology", "group_size", "dim_param", "num_dims",
                      "count", "batch_param", "model_config_dim"]
COMM_COLUMNS = ["operation", "tp_size", "size_bytes", "dtype", "dim_param", "num_dims", "backend", "latency_ms", "run"]
INTEGER = {"num_tokens", "num_requests", "is_context", "is_prefill", "kv_cache_size", "prefill_chunk_size",
           "activated_experts", "local_pairs", "num_tensor_parallel_workers", "count", "group_size", "dim_param",
           "num_dims", "model_config_dim", "tp_size", "size_bytes"}
FLOAT = set(STATS) | {"latency_ms"}


def _name(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-") or "unnamed"


def deployment_name(tp, ep, dtype, kv_cache_dtype):
    return f"tp{tp}" + (f"-ep{ep}" if ep and ep != 1 else "") + f"-{dtype}-kv{kv_cache_dtype}"


def _cell(value):
    return "" if value is None else value


def _value(column, text):
    if text == "":
        return None
    if column in INTEGER:
        return int(text)
    if column in FLOAT:
        return float(text)
    return text


def _write_csv(path, columns, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(columns)
        for row in rows:
            w.writerow([_cell(row[c]) for c in columns])


def _read_csv(path):
    with open(path, newline="") as f:
        return [{k: _value(k, v) for k, v in row.items()} for row in csv.DictReader(f)]


def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def _run_entry(row):
    return {"run_uid": row["run_uid"], "started_at": row["started_at"], "device": row["device"],
            "gpu_name": row["gpu_name"], "engine": row["engine"], "backend": row["backend"],
            "stack": json.loads(row["stack_json"]) if row["stack_json"] else {}}


def _order(columns):
    # NULLs first, then by value, so the files come out in the same order from every database
    return ", ".join(f"{c} IS NOT NULL, {c}" for c in columns)


# --------------------------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------------------------

def export_db(db_path, out_dir):
    """Write ``db_path``'s measurements under ``out_dir``; returns counts for a summary."""
    con = sqlite3.connect(f"file:{os.path.abspath(db_path)}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    cols = {t: {r[1] for r in con.execute(f"PRAGMA table_info({t})")}
            for t in ("profiled_models", "results", "coll_op_results", "profile_runs")}
    labelled = "device" in cols["profiled_models"] and "device" in cols["results"]
    runs = {r["run_id"]: r for r in con.execute("SELECT * FROM profile_runs")} if cols["profile_runs"] else {}
    done = {"deployments": 0, "operation_files": 0, "result_rows": 0, "communication_files": 0,
            "communication_rows": 0, "skipped_deployments": 0, "skipped_communication_rows": 0, "duplicate_rows": 0}
    counted = set()   # (signature, device, engine) whose second measurements are counted

    models = con.execute("SELECT * FROM profiled_models ORDER BY model_id").fetchall() if cols["profiled_models"] else []
    for pm in models:
        if not labelled or pm["device"] is None or pm["engine"] is None:
            done["skipped_deployments"] += 1
            continue
        folder = os.path.join(out_dir, "compute", pm["device"], pm["engine"], pm["backend_version"] or pm["backend"],
                              *pm["model_name"].split("/"),
                              deployment_name(pm["tp_degree"], pm["ep_degree"], pm["dtype"], pm["kv_cache_dtype"]))
        os.makedirs(folder, exist_ok=True)
        ops = con.execute("""SELECT s.signature_hash, s.operation_name, s.signature_json, mo.count
                             FROM model_operations mo JOIN signatures s ON s.operation_id = mo.operation_id
                             WHERE mo.model_id = ? ORDER BY s.operation_name, s.signature_hash""",
                          (pm["model_id"],)).fetchall()
        model_rows, used_runs, files = [], set(), set()
        for op in ops:
            key = (op["signature_hash"], pm["device"], pm["engine"])
            # One row per batch shape: the first measurement the database got, should it have several
            rows = con.execute(f"""SELECT * FROM results WHERE id IN (
                                       SELECT MIN(id) FROM results WHERE signature_hash = ? AND device = ? AND engine = ?
                                       GROUP BY {', '.join(BATCH)})
                                   ORDER BY {_order(BATCH)}""", key).fetchall()
            if key not in counted:
                counted.add(key)
                done["duplicate_rows"] += con.execute(
                    "SELECT COUNT(*) FROM results WHERE signature_hash = ? AND device = ? AND engine = ?",
                    key).fetchone()[0] - len(rows)
            fname = f"{_name(op['operation_name'])}-{op['signature_hash'][:8]}.csv" if rows else ""
            if rows:
                out = []
                for r in rows:
                    run = runs.get(r["run_id"])
                    if run is not None:
                        used_runs.add(r["run_id"])
                    out.append({**{c: r[c] for c in SHAPE + STATS + ["input_shapes", "created_at"]},
                                "run": run["run_uid"] if run is not None else None})
                _write_csv(os.path.join(folder, fname), RESULT_COLUMNS, out)
                files.add(fname)
                done["operation_files"] += 1
                done["result_rows"] += len(rows)
            model_rows.append({"file": fname, "operation_name": op["operation_name"],
                               "signature_hash": op["signature_hash"], "count": op["count"],
                               "signature_json": op["signature_json"]})
        _write_csv(os.path.join(folder, "model.csv"), MODEL_COLUMNS, model_rows)
        colls = con.execute("""SELECT co.signature_hash, co.operation_name, co.topology, co.tp_degree AS group_size,
                                      co.dim_param, co.num_dims, mc.count, mcc.batch_param, mcc.model_config_dim
                               FROM model_collectives mc JOIN coll_ops co ON co.collective_id = mc.collective_id
                               LEFT JOIN model_collective_configs mcc
                                    ON mcc.model_id = mc.model_id AND mcc.collective_id = mc.collective_id
                               WHERE mc.model_id = ? ORDER BY co.operation_name, co.signature_hash""",
                            (pm["model_id"],)).fetchall()
        _write_csv(os.path.join(folder, "collectives.csv"), COLLECTIVE_COLUMNS, colls)
        _write_json(os.path.join(folder, "deployment.json"), {
            "model": pm["model_name"], "model_family": pm["model_family"], "backend": pm["backend"],
            "backend_version": pm["backend_version"], "tp_degree": pm["tp_degree"], "ep_degree": pm["ep_degree"],
            "dtype": pm["dtype"], "kv_cache_dtype": pm["kv_cache_dtype"], "device": pm["device"],
            "engine": pm["engine"], "created_at": pm["created_at"],
            "runs": sorted((_run_entry(runs[i]) for i in used_runs), key=lambda r: (r["started_at"], r["run_uid"]))})
        for stale in os.listdir(folder):   # operation files a previous export left that this model no longer has
            if stale.endswith(".csv") and stale not in files | {"model.csv", "collectives.csv"}:
                os.remove(os.path.join(folder, stale))
        done["deployments"] += 1

    if "engine" not in cols["coll_op_results"]:   # collectives measured before engines were recorded
        done["skipped_communication_rows"] = (con.execute("SELECT COUNT(*) FROM coll_op_results").fetchone()[0]
                                              if cols["coll_op_results"] else 0)
        con.close()
        return done
    done["skipped_communication_rows"] = con.execute(
        "SELECT COUNT(*) FROM coll_op_results WHERE engine IS NULL").fetchone()[0]
    groups = con.execute("""SELECT DISTINCT topology, engine, operation FROM coll_op_results
                            WHERE engine IS NOT NULL ORDER BY topology, engine, operation""").fetchall()
    comm_runs = {}
    for g in groups:
        folder = os.path.join(out_dir, "communication", g["topology"], g["engine"])
        os.makedirs(folder, exist_ok=True)
        rows = con.execute(f"""SELECT * FROM coll_op_results WHERE topology = ? AND engine = ? AND operation = ?
                               ORDER BY {_order(['tp_size', 'dim_param', 'num_dims', 'dtype', 'backend', 'size_bytes'])}""",
                           (g["topology"], g["engine"], g["operation"])).fetchall()
        out = []
        for r in rows:
            run = runs.get(r["run_id"])
            if run is not None:
                comm_runs.setdefault(folder, set()).add(r["run_id"])
            out.append({**{c: r[c] for c in COMM_COLUMNS if c != "run"}, "run": run["run_uid"] if run is not None else None})
        _write_csv(os.path.join(folder, f"{_name(g['operation'])}.csv"), COMM_COLUMNS, out)
        done["communication_files"] += 1
        done["communication_rows"] += len(rows)
    for folder in {os.path.join(out_dir, "communication", g["topology"], g["engine"]) for g in groups}:
        _write_json(os.path.join(folder, "runs.json"), {"runs": sorted(
            (_run_entry(runs[i]) for i in comm_runs.get(folder, ())), key=lambda r: (r["started_at"], r["run_uid"]))})
    con.close()
    return done


# --------------------------------------------------------------------------------------------
# Import
# --------------------------------------------------------------------------------------------

def _runs(con, entries):
    """{run_uid: run_id in this database}, adding the runs it does not have."""
    ids = {}
    for e in entries:
        stack = dict(e.get("stack") or {})
        stack.update({k: e.get(k) for k in ("device", "gpu_name", "engine", "backend")}, run_uid=e["run_uid"])
        ids[e["run_uid"]] = profile_db.record_run(con, stack, started_at=e.get("started_at"))
    return ids


def _operation_id(con, signature_hash, operation_name, signature_json):
    row = con.execute("SELECT operation_id FROM signatures WHERE signature_hash = ?", (signature_hash,)).fetchone()
    if row:
        return row[0]
    return con.execute("INSERT INTO signatures (signature_hash, operation_name, signature_json) VALUES (?, ?, ?)",
                       (signature_hash, operation_name, signature_json)).lastrowid


def _import_deployment(con, folder, done):
    with open(os.path.join(folder, "deployment.json")) as f:
        d = json.load(f)
    run_ids = _runs(con, d.get("runs", []))
    key = (d["model"], d["model_family"], d["backend"], d["tp_degree"], d["dtype"], d["kv_cache_dtype"],
           d["ep_degree"], d["device"], d["engine"])
    row = con.execute("""SELECT model_id FROM profiled_models WHERE model_name = ? AND model_family = ? AND backend = ?
                         AND tp_degree = ? AND dtype = ? AND kv_cache_dtype = ? AND ep_degree = ?
                         AND device IS ? AND engine IS ?""", key).fetchone()
    if row:
        model_id = row[0]
    else:
        model_id = con.execute("""INSERT INTO profiled_models (model_name, model_family, backend, tp_degree, dtype,
                                      kv_cache_dtype, ep_degree, device, engine, backend_version, created_at)
                                  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                               key + (d.get("backend_version"), d.get("created_at"))).lastrowid
        done["deployments_added"] += 1
    for op in _read_csv(os.path.join(folder, "model.csv")):
        operation_id = _operation_id(con, op["signature_hash"], op["operation_name"], op["signature_json"])
        if con.execute("SELECT 1 FROM model_operations WHERE model_id = ? AND operation_id = ?",
                       (model_id, operation_id)).fetchone():
            con.execute("UPDATE model_operations SET count = ? WHERE model_id = ? AND operation_id = ?",
                        (op["count"], model_id, operation_id))
        else:
            con.execute("INSERT INTO model_operations (model_id, operation_id, count) VALUES (?, ?, ?)",
                        (model_id, operation_id, op["count"]))
        if not op["file"]:
            continue
        for r in _read_csv(os.path.join(folder, op["file"])):
            # Already measured at this batch shape, by a run of any TP degree
            where = " AND ".join(f"{c} IS ?" for c in BATCH)
            if con.execute(f"SELECT 1 FROM results WHERE signature_hash = ? AND {where} AND device IS ? AND engine IS ?",
                           [op["signature_hash"]] + [r[c] for c in BATCH] + [d["device"], d["engine"]]).fetchone():
                done["result_rows_present"] += 1
                continue
            cols = ["signature_hash"] + SHAPE + STATS + ["input_shapes", "created_at", "device", "engine", "run_id"]
            vals = ([op["signature_hash"]] + [r[c] for c in SHAPE + STATS + ["input_shapes", "created_at"]]
                    + [d["device"], d["engine"], run_ids.get(r["run"])])
            con.execute(f"INSERT INTO results ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", vals)
            done["result_rows_added"] += 1
    for c in _read_csv(os.path.join(folder, "collectives.csv")):
        row = con.execute("SELECT collective_id FROM coll_ops WHERE signature_hash = ?", (c["signature_hash"],)).fetchone()
        collective_id = row[0] if row else con.execute(
            """INSERT INTO coll_ops (signature_hash, operation_name, topology, tp_degree, num_dims, dim_param)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (c["signature_hash"], c["operation_name"], c["topology"], c["group_size"], c["num_dims"],
             c["dim_param"])).lastrowid
        con.execute("""INSERT INTO model_collectives (model_id, collective_id, count) VALUES (?, ?, ?)
                       ON CONFLICT(model_id, collective_id) DO UPDATE SET count = excluded.count""",
                    (model_id, collective_id, c["count"]))
        if c["batch_param"] is not None:
            con.execute("""INSERT OR IGNORE INTO model_collective_configs (model_id, collective_id, batch_param,
                               model_config_dim) VALUES (?, ?, ?, ?)""",
                        (model_id, collective_id, c["batch_param"], c["model_config_dim"]))


def _import_communication(con, folder, topology, engine, done):
    runs_file = os.path.join(folder, "runs.json")
    run_ids = {}
    if os.path.exists(runs_file):
        with open(runs_file) as f:
            run_ids = _runs(con, json.load(f).get("runs", []))
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".csv"):
            continue
        for r in _read_csv(os.path.join(folder, fname)):
            key = (r["operation"], topology, r["backend"], r["dtype"], r["tp_size"], r["size_bytes"], r["dim_param"],
                   r["num_dims"], engine)
            # IS, not = or the table's UNIQUE key: dim_param is NULL for some collectives
            if con.execute("""SELECT 1 FROM coll_op_results WHERE operation IS ? AND topology IS ? AND backend IS ?
                              AND dtype IS ? AND tp_size IS ? AND size_bytes IS ? AND dim_param IS ? AND num_dims IS ?
                              AND engine IS ?""", key).fetchone():
                done["communication_rows_present"] += 1
                continue
            con.execute("""INSERT INTO coll_op_results (operation, topology, backend, tp_size, size_bytes, dtype,
                               dim_param, num_dims, latency_ms, engine, run_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (r["operation"], topology, r["backend"], r["tp_size"], r["size_bytes"], r["dtype"],
                         r["dim_param"], r["num_dims"], r["latency_ms"], engine, run_ids.get(r["run"])))
            done["communication_rows_added"] += 1


def import_tree(tree_dir, db_path):
    """Merge the tree under ``tree_dir`` (as export_db() writes it) into ``db_path``."""
    con = sqlite3.connect(db_path)
    profile_db.init_schema(con)
    done = {"deployments": 0, "deployments_added": 0, "result_rows_added": 0, "result_rows_present": 0,
            "communication_rows_added": 0, "communication_rows_present": 0}
    for root, _dirs, files in sorted(os.walk(os.path.join(tree_dir, "compute"))):
        if "deployment.json" in files:
            _import_deployment(con, root, done)
            done["deployments"] += 1
    comm = os.path.join(tree_dir, "communication")
    for topology in sorted(os.listdir(comm)) if os.path.isdir(comm) else []:
        for engine in sorted(os.listdir(os.path.join(comm, topology))):
            folder = os.path.join(comm, topology, engine)
            if os.path.isdir(folder):
                _import_communication(con, folder, topology, engine, done)
    con.commit()
    con.close()
    return done


# --------------------------------------------------------------------------------------------
# Verify
# --------------------------------------------------------------------------------------------

def verify_tree(tree_dir):
    """(problems, counts): what is wrong with the tree under ``tree_dir`` (an empty list when it
    is valid). Valid means it imports into an empty database and exporting that database gives
    back exactly the files under compute/ and communication/: every file parses, every value is
    as export writes it, nothing is missing and nothing else is there. Other files at the top
    (a README) are not checked."""
    import filecmp
    import tempfile
    missing = []   # files a model.csv lists that are not there, or that measure a batch shape twice
    for root, _dirs, files in sorted(os.walk(os.path.join(tree_dir, "compute"))):
        if "model.csv" not in files:
            continue
        rel = os.path.relpath(root, tree_dir)
        try:
            listed = [r["file"] for r in _read_csv(os.path.join(root, "model.csv")) if r["file"]]
        except Exception as e:  # noqa: BLE001
            missing.append(f"{rel}/model.csv: unreadable ({type(e).__name__}: {e})")
            continue
        for f in listed:
            if not os.path.exists(os.path.join(root, f)):
                missing.append(f"{rel}/{f}: missing (model.csv lists it)")
                continue
            try:
                shapes = [tuple(r[c] for c in BATCH) for r in _read_csv(os.path.join(root, f))]
            except Exception:  # noqa: BLE001 -- malformed: the round trip below reports it
                continue
            if len(set(shapes)) < len(shapes):
                missing.append(f"{rel}/{f}: {len(shapes) - len(set(shapes))} batch shape(s) measured more than once "
                               f"(e.g. by runs of different TP degrees); dooly export writes one row per batch shape")
    if missing:
        return missing, {}
    with tempfile.TemporaryDirectory() as tmp:
        try:
            counts = import_tree(tree_dir, os.path.join(tmp, "check.db"))
        except Exception as e:  # noqa: BLE001 -- a malformed file: report it, do not crash
            return [f"does not import: {type(e).__name__}: {e}"], {}
        export_db(os.path.join(tmp, "check.db"), os.path.join(tmp, "export"))
        problems = []
        for sub in ("compute", "communication"):
            ours, theirs = os.path.join(tmp, "export", sub), os.path.join(tree_dir, sub)
            mine = {os.path.relpath(os.path.join(r, f), ours) for r, _, fs in os.walk(ours) for f in fs}
            given = {os.path.relpath(os.path.join(r, f), theirs) for r, _, fs in os.walk(theirs) for f in fs}
            for rel in sorted(given - mine):
                problems.append(f"{sub}/{rel}: not written by dooly export (a stray file, or a deployment without "
                                f"GPU and engine labels)")
            for rel in sorted(mine - given):
                problems.append(f"{sub}/{rel}: missing (its deployment references it)")
            for rel in sorted(mine & given):
                if not filecmp.cmp(os.path.join(ours, rel), os.path.join(theirs, rel), shallow=False):
                    problems.append(f"{sub}/{rel}: differs from what dooly export writes for the same data "
                                    f"(edited by hand, or two exports disagree)")
        return problems, counts
