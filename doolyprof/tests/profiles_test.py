"""The profile database's device/engine labels and ``dooly export`` / ``dooly import``.

- export -> import into an empty database -> export gives byte-identical files
- import is a union: importing twice adds nothing; rows of another device are kept apart
- the profiler's reuse lookup sees only rows of its own device and engine
- an older database (no device, engine or run columns) is brought up to date, keeps its rows,
  and its unlabelled rows are neither reused nor exported

Runs directly: ``python doolyprof/tests/profiles_test.py``. CPU only, standard library only.
"""
import filecmp
import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from doolyprof.profiler import db as profile_db  # noqa: E402
from doolyprof import profiles  # noqa: E402


def _stack(device, engine="vllm-0.33.0", uid=None):
    return {"run_uid": uid or f"run-{device}-{engine}", "device": device, "gpu_name": device.upper(), "engine": engine,
            "backend": "FLASH_ATTN-fa2", "versions": {"vllm": engine.split("-")[-1]}, "argv": ["run-profiler.py"]}


def _fill(path, device, engine="vllm-0.33.0", model="org/model-a", sigs=("s1", "s2"), latency=1.0):
    """A database with one deployment of ``model`` measured on ``device`` (the rows the profiler writes)."""
    con = sqlite3.connect(path)
    profile_db.init_schema(con)
    stack = _stack(device, engine)
    run_id = profile_db.record_run(con, stack, started_at="2026-11-20 10:00:00")
    model_id = con.execute("""INSERT INTO profiled_models (model_name, model_family, backend, tp_degree, dtype,
                                  kv_cache_dtype, ep_degree, device, engine, backend_version, created_at)
                              VALUES (?, 'fam', 'FLASH_ATTN', 2, 'bfloat16', 'auto', 1, ?, ?, 'FLASH_ATTN-fa2',
                                      '2026-11-20 10:00:01')""", (model, device, engine)).lastrowid
    for i, sig in enumerate(sigs):
        op = con.execute("SELECT operation_id FROM signatures WHERE signature_hash = ?", (sig,)).fetchone()
        op = op[0] if op else con.execute("INSERT INTO signatures (signature_hash, operation_name, signature_json) "
                                          "VALUES (?, ?, ?)", (sig, "aten::linear" if i else "Attention",
                                                               f'["{sig}", [["input1_dim1", 4096]]]')).lastrowid
        con.execute("INSERT INTO model_operations (model_id, operation_id, count) VALUES (?, ?, ?)", (model_id, op, 32))
        for n in (1, 2, 128):
            con.execute("""INSERT INTO results (signature_hash, mean_latency_ms, median_latency_ms, max_latency_ms,
                               min_latency_ms, std_latency_ms, num_tokens, num_requests, is_context, is_prefill,
                               kv_cache_size, num_tensor_parallel_workers, input_shapes, created_at, device, engine, run_id)
                           VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, 2, '[[1, 4096]]', '2026-11-20 10:00:02', ?, ?, ?)""",
                        (sig, latency * n + 0.1, latency * n, latency * n * 1.5, latency * n * 0.9, 1e-3 / 3,
                         n, 1 if i == 0 else 0, 1 if i == 0 else None, 64 if i == 0 else None, device, engine, run_id))
    co = con.execute("""INSERT OR IGNORE INTO coll_ops (signature_hash, operation_name, topology, tp_degree, num_dims,
                            dim_param) VALUES ('c1', 'all_reduce', ?, 2, 2, -1)""", (device + "-pcie",)).lastrowid
    co = con.execute("SELECT collective_id FROM coll_ops WHERE signature_hash = 'c1'").fetchone()[0]
    con.execute("INSERT INTO model_collectives (model_id, collective_id, count) VALUES (?, ?, 65)", (model_id, co))
    con.execute("""INSERT INTO model_collective_configs (model_id, collective_id, batch_param, model_config_dim)
                   VALUES (?, ?, 'NUM_TOKS', 4096)""", (model_id, co))
    for size in (1024, 2048):   # dim_param -1 for all_gather, NULL for all_reduce (as the profiler stores them)
        for op, dim in (("all_gather", -1), ("all_reduce", None)):
            con.execute("""INSERT INTO coll_op_results (operation, topology, backend, tp_size, size_bytes, dtype,
                               dim_param, num_dims, latency_ms, engine, run_id)
                           VALUES (?, ?, 'NCCL', 2, ?, 'bfloat16', ?, 2, ?, ?, ?)""",
                        (op, device + "-pcie", size, dim, size / 1e5, engine, run_id))
    con.commit()
    con.close()


def _same_tree(a, b):
    cmp = filecmp.dircmp(a, b)
    stack = [cmp]
    while stack:
        c = stack.pop()
        assert not (c.left_only or c.right_only), (c.left, c.left_only, c.right_only)
        _, mismatch, errors = filecmp.cmpfiles(c.left, c.right, c.common_files, shallow=False)
        assert not mismatch and not errors, (c.left, mismatch, errors)
        stack.extend(c.subdirs.values())


def test_export_import_round_trip():
    with tempfile.TemporaryDirectory() as d:
        _fill(f"{d}/a.db", "gpu-x")
        done = profiles.export_db(f"{d}/a.db", f"{d}/t1")
        assert done["deployments"] == 1 and done["operation_files"] == 2 and done["result_rows"] == 6, done
        folder = f"{d}/t1/compute/gpu-x/vllm-0.33.0/FLASH_ATTN-fa2/org/model-a/tp2-bfloat16-kvauto"
        assert sorted(os.listdir(folder)) == ["Attention-s1.csv", "aten-linear-s2.csv", "collectives.csv",
                                              "deployment.json", "model.csv"], os.listdir(folder)
        assert os.listdir(f"{d}/t1/communication/gpu-x-pcie/vllm-0.33.0") != []
        profiles.import_tree(f"{d}/t1", f"{d}/b.db")
        profiles.export_db(f"{d}/b.db", f"{d}/t2")
        _same_tree(f"{d}/t1", f"{d}/t2")


def test_import_is_a_union_and_keeps_devices_apart():
    with tempfile.TemporaryDirectory() as d:
        _fill(f"{d}/x.db", "gpu-x")
        _fill(f"{d}/y.db", "gpu-y", latency=2.0)
        profiles.export_db(f"{d}/y.db", f"{d}/ty")
        first = profiles.import_tree(f"{d}/ty", f"{d}/x.db")
        again = profiles.import_tree(f"{d}/ty", f"{d}/x.db")
        assert first["result_rows_added"] == 6 and first["deployments_added"] == 1, first
        assert again["result_rows_added"] == 0 and again["result_rows_present"] == 6, again
        assert first["communication_rows_added"] == 4 and again["communication_rows_added"] == 0, (first, again)
        con = sqlite3.connect(f"{d}/x.db")
        per_device = dict(con.execute("SELECT device, COUNT(*) FROM results GROUP BY device").fetchall())
        assert per_device == {"gpu-x": 6, "gpu-y": 6}, per_device
        assert con.execute("SELECT COUNT(*) FROM profiled_models").fetchone()[0] == 2
        assert con.execute("SELECT COUNT(*) FROM signatures").fetchone()[0] == 2   # shared by both devices
        y = con.execute("SELECT mean_latency_ms FROM results WHERE device = 'gpu-y' AND num_tokens = 128 "
                        "AND signature_hash = 's2'").fetchone()[0]
        assert y == 2.0 * 128 + 0.1, y


def test_reuse_only_for_the_same_device_and_engine():
    from doolyprof.profiler.module import ModuleComparator   # imports torch: skip without it
    with tempfile.TemporaryDirectory() as d:
        _fill(f"{d}/x.db", "gpu-x")
        c = ModuleComparator()
        assert c._load_existing_signatures(f"{d}/x.db", "gpu-x", "vllm-0.33.0") == {"s1", "s2"}
        assert c._load_existing_signatures(f"{d}/x.db", "gpu-y", "vllm-0.33.0") == set()
        assert c._load_existing_signatures(f"{d}/x.db", "gpu-x", "vllm-0.34.0") == set()


def test_older_database_is_upgraded_not_reused_not_exported():
    with tempfile.TemporaryDirectory() as d:
        con = sqlite3.connect(f"{d}/old.db")   # the tables as DoolyProf created them before device labels
        con.executescript("""
            CREATE TABLE signatures (operation_id INTEGER PRIMARY KEY AUTOINCREMENT, signature_hash TEXT UNIQUE NOT NULL,
                operation_name TEXT NOT NULL, signature_json TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE results (id INTEGER PRIMARY KEY AUTOINCREMENT, signature_hash TEXT NOT NULL, mean_latency_ms REAL,
                median_latency_ms REAL, max_latency_ms REAL, min_latency_ms REAL, std_latency_ms REAL, num_tokens INTEGER,
                num_requests INTEGER, is_context BOOLEAN NULL, is_prefill BOOLEAN NULL, kv_cache_size INTEGER NULL,
                prefill_chunk_size INTEGER NULL, activated_experts INTEGER NULL, num_tensor_parallel_workers INTEGER,
                input_shapes TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, local_pairs INTEGER NULL);
            CREATE TABLE profiled_models (model_id INTEGER PRIMARY KEY AUTOINCREMENT, model_name TEXT NOT NULL,
                model_family TEXT NOT NULL, backend TEXT NOT NULL, tp_degree INTEGER NOT NULL, dtype TEXT NOT NULL,
                kv_cache_dtype TEXT NOT NULL DEFAULT 'auto', created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ep_degree INTEGER NOT NULL DEFAULT 1);
            CREATE TABLE model_operations (id INTEGER PRIMARY KEY AUTOINCREMENT, model_id INTEGER NOT NULL,
                operation_id INTEGER NOT NULL, count INTEGER NOT NULL DEFAULT 1, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(model_id, operation_id, count));
            CREATE TABLE coll_op_results (id INTEGER PRIMARY KEY AUTOINCREMENT, operation TEXT NOT NULL, topology TEXT NOT NULL,
                backend TEXT NOT NULL, tp_size INTEGER NOT NULL, size_bytes INTEGER NOT NULL, dtype TEXT, dim_param INTEGER,
                num_dims INTEGER, latency_ms REAL NOT NULL,
                UNIQUE(operation, topology, backend, dtype, tp_size, size_bytes, dim_param, num_dims));
            INSERT INTO signatures (signature_hash, operation_name) VALUES ('old', 'aten::linear');
            INSERT INTO results (signature_hash, mean_latency_ms, num_tokens, num_tensor_parallel_workers) VALUES ('old', 1.5, 8, 1);
            INSERT INTO profiled_models (model_name, model_family, backend, tp_degree, dtype) VALUES ('m', 'f', 'FLASH_ATTN', 1, 'bfloat16');
            INSERT INTO model_operations (model_id, operation_id, count) VALUES (1, 1, 3);
            INSERT INTO coll_op_results (operation, topology, backend, tp_size, size_bytes, dtype, dim_param, num_dims, latency_ms)
                VALUES ('all_reduce', 't', 'NCCL', 2, 1024, 'bfloat16', -1, 2, 0.25);
        """)
        con.close()
        done = profiles.export_db(f"{d}/old.db", f"{d}/t")   # read-only: exports nothing, changes nothing
        assert done["deployments"] == 0 and done["skipped_deployments"] == 1 and done["skipped_communication_rows"] == 1, done
        con = sqlite3.connect(f"{d}/old.db")
        assert "device" not in [r[1] for r in con.execute("PRAGMA table_info(results)")]
        profile_db.init_schema(con)
        assert con.execute("SELECT mean_latency_ms, device, engine FROM results").fetchall() == [(1.5, None, None)]
        assert con.execute("SELECT latency_ms, engine FROM coll_op_results").fetchall() == [(0.25, None)]
        con.close()
        try:
            from doolyprof.profiler.module import ModuleComparator
        except ImportError:
            return
        assert ModuleComparator()._load_existing_signatures(f"{d}/old.db", "gpu-x", "vllm-0.33.0") == set()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except ImportError as e:
                print("SKIP", name, e)
                continue
            print("PASS", name)
