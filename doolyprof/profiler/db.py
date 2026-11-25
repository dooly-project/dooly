"""The profile database (profile.db): its tables, created or brought up to date by
``init_schema()``, and the labels of the hardware and software a run measures on.

Every latency row is stored with the GPU it was measured on (``device``, see
``doolyprof.profiler.utils.topology.device_label``) and the serving engine and version it ran
under (``engine``, e.g. ``vllm-0.33.0``); collective measurements carry the engine next to
their topology label. A measurement is reused only for the same device and engine (topology
and engine for collectives), so a database can hold results of several GPUs and engine
versions, e.g. after ``dooly import``. Rows written before these columns existed have neither
and are never reused.

Each profiling run is recorded in ``profile_runs``, with a ``run_uid`` that stays the same in
every database the run's rows are exported to or imported into, and the versions it ran with
(``stack_json``); the rows it wrote carry its ``run_id``. ``profiled_models`` rows also carry
the device, the engine and the attention backend's version (``backend_version``, e.g.
``FLASHINFER-0.7.2.post1``).

Only the standard library is imported, so ``dooly export`` / ``dooly import`` and the tests
can use this module without PyTorch or vLLM.
"""
import json
import os
import platform
import re
import sqlite3
import subprocess
import uuid
from importlib import metadata

# Distributions whose versions a run records (the ones installed; the rest are skipped).
RECORDED_DISTRIBUTIONS = ("vllm", "torch", "transformers", "triton", "flashinfer-python",
                          "nvidia-nccl-cu13", "nvidia-nccl-cu12", "nvidia-cuda-runtime",
                          "nvidia-cuda-runtime-cu12", "numpy")

# The library that implements an attention backend, for backend labels: the first entry whose
# key is part of the backend name. FLASH_ATTN backends are labelled with the FlashAttention
# version vLLM is asked to use instead; backends not listed keep their name alone.
BACKEND_LIBRARIES = (("FLASHINFER", "flashinfer-python"), ("TRITON", "triton"))


# --------------------------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------------------------

def _version(dist):
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def engine_label():
    """DOOLY_ENGINE if set, else the serving engine and its version, e.g. ``vllm-0.33.0``."""
    if os.environ.get("DOOLY_ENGINE"):
        return os.environ["DOOLY_ENGINE"]
    return f"vllm-{_version('vllm') or 'unknown'}"


def backend_label(backend, flash_attn_version=None):
    """The attention backend with the version of what implements it, e.g. ``FLASH_ATTN-fa2``,
    ``FLASHINFER-0.7.2.post1``, ``TRITON_ATTN-3.8.0``; the name alone when unknown."""
    name = str(backend)
    if "FLASH_ATTN" in name and flash_attn_version:
        return f"{name}-fa{flash_attn_version}"
    for part, dist in BACKEND_LIBRARIES:
        if part in name:
            version = _version(dist)
            return f"{name}-{version}" if version else name
    return name


def _driver_version():
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            v = pynvml.nvmlSystemGetDriverVersion()
            return v.decode() if isinstance(v, bytes) else v
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001 -- no NVML
        return None


def _revision():
    """DOOLY_REVISION if set, else the git commit DoolyProf runs from, else None."""
    if os.environ.get("DOOLY_REVISION"):
        return os.environ["DOOLY_REVISION"]
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:  # noqa: BLE001 -- no git
        return None


def portable_argv(argv, home=None):
    """``argv`` with the home directory (default: this user's) written as ``~`` wherever a path
    starts with it (an argument, a ``--opt=`` value, an item of a list): the command line is
    exported with the run (deployment.json, shared through DoolyDB), and the rest of a path still
    says what was read and written."""
    home = (os.path.expanduser("~") if home is None else home).rstrip("/")
    if len(home) < 2:  # no home directory, or "/"
        return list(argv)
    pattern = re.compile(rf"(?<![^\s=:,]){re.escape(home)}(?=/|$|[\s:,])")
    return [pattern.sub("~", arg) for arg in argv]


def run_stack(backend, flash_attn_version=None, argv=None):
    """What a profiling run records about itself: its labels (device, engine, backend), the GPU
    model, driver and package versions, the DoolyProf commit and the command line (home directory
    as ``~``, ``portable_argv``)."""
    from .utils.topology import device_label, gpu_name
    return {
        "run_uid": uuid.uuid4().hex,
        "device": device_label(),
        "gpu_name": gpu_name(),
        "engine": engine_label(),
        "backend": backend_label(backend, flash_attn_version),
        "driver": _driver_version(),
        "python": platform.python_version(),
        "versions": {d: v for d in RECORDED_DISTRIBUTIONS if (v := _version(d))},
        "doolyprof_revision": _revision(),
        "argv": portable_argv(argv) if argv else None,
    }


# --------------------------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------------------------

def _add_column(conn, table, definition):
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {definition}")
    except sqlite3.OperationalError:
        pass  # column already present


def _columns(conn, table):
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


COLL_OP_RESULTS = """
    CREATE TABLE coll_op_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        operation TEXT NOT NULL,
        topology TEXT NOT NULL,
        backend TEXT NOT NULL,
        tp_size INTEGER NOT NULL,
        size_bytes INTEGER NOT NULL,
        dtype TEXT,
        dim_param INTEGER,
        num_dims INTEGER,
        latency_ms REAL NOT NULL,
        engine TEXT,
        run_id INTEGER,
        UNIQUE(operation, topology, backend, dtype, tp_size, size_bytes, dim_param, num_dims, engine)
    )
"""


def _coll_op_results(conn):
    """coll_op_results with the engine in its key. An older table (no engine column, a key
    without it) is rebuilt once with its rows copied; they keep no engine."""
    cols = _columns(conn, "coll_op_results")
    if not cols:
        conn.execute(COLL_OP_RESULTS)
    elif "engine" not in cols:
        conn.execute("ALTER TABLE coll_op_results RENAME TO coll_op_results_before_engine")
        conn.execute(COLL_OP_RESULTS)
        old = ", ".join(c for c in cols)
        conn.execute(f"INSERT INTO coll_op_results ({old}) SELECT {old} FROM coll_op_results_before_engine")
        conn.execute("DROP TABLE coll_op_results_before_engine")


def init_schema(conn):
    """Create every table of the profile database, or bring an older database up to date."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signatures (
            operation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            signature_hash TEXT UNIQUE NOT NULL,
            operation_name TEXT NOT NULL,
            signature_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signature_hash TEXT NOT NULL,
            mean_latency_ms REAL,
            median_latency_ms REAL,
            max_latency_ms REAL,
            min_latency_ms REAL,
            std_latency_ms REAL,
            num_tokens INTEGER,
            num_requests INTEGER,
            is_context BOOLEAN NULL,
            is_prefill BOOLEAN NULL,
            kv_cache_size INTEGER NULL,
            prefill_chunk_size INTEGER NULL,
            activated_experts INTEGER NULL,
            num_tensor_parallel_workers INTEGER,
            input_shapes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (signature_hash) REFERENCES signatures(signature_hash)
        )
    """)
    # Columns added after the table first shipped (CREATE TABLE IF NOT EXISTS won't add them to
    # an existing table): activated_experts; local_pairs (expert-parallel MoE rows: pairs routed
    # to this rank's experts, NULL on every other row); device, engine and run_id (NULL on rows
    # written before they existed).
    for definition in ("activated_experts INTEGER NULL", "local_pairs INTEGER NULL",
                       "device TEXT", "engine TEXT", "run_id INTEGER"):
        _add_column(conn, "results", definition)
    # Composite index covering the per-row "does this exact (sig, workload params, device,
    # engine) row already exist?" lookup of the result writer; without it that SELECT is a full
    # scan, making the write phase O(N^2) in result rows.
    conn.execute("DROP INDEX IF EXISTS idx_results_lookup")
    conn.execute("DROP INDEX IF EXISTS idx_results_lookup_v2")
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_results_lookup_v3
        ON results (signature_hash, num_tokens, num_requests,
                    is_context, is_prefill, kv_cache_size,
                    prefill_chunk_size, activated_experts, local_pairs,
                    num_tensor_parallel_workers, device, engine)
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS profiled_models (
            model_id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            model_family TEXT NOT NULL,
            backend TEXT NOT NULL,
            tp_degree INTEGER NOT NULL,
            dtype TEXT NOT NULL,
            kv_cache_dtype TEXT NOT NULL DEFAULT 'auto',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Older databases lack kv_cache_dtype (every row in them was profiled with vLLM's default,
    # "auto") and ep_degree (rows written before fake EP had experts TP-sharded, ep 1); device,
    # engine and backend_version are NULL on rows written before they existed.
    for definition in ("kv_cache_dtype TEXT NOT NULL DEFAULT 'auto'", "ep_degree INTEGER NOT NULL DEFAULT 1",
                       "device TEXT", "engine TEXT", "backend_version TEXT"):
        _add_column(conn, "profiled_models", definition)
    # One row per deployment: model, backend, tp, dtype, KV-cache dtype, ep, device and engine.
    # An index rather than an inline UNIQUE constraint, so it can be added to existing
    # databases without recreating the table; earlier versions of it are dropped.
    for old in ("ix_profiled_models_unique", "ix_profiled_models_unique_v2", "ix_profiled_models_unique_v3"):
        conn.execute(f"DROP INDEX IF EXISTS {old}")
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS ix_profiled_models_unique_v4
            ON profiled_models(model_name, backend, tp_degree, dtype, kv_cache_dtype, ep_degree, device, engine)
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS model_operations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id INTEGER NOT NULL,
            operation_id INTEGER NOT NULL,
            count INTEGER NOT NULL DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (model_id) REFERENCES profiled_models(model_id),
            FOREIGN KEY (operation_id) REFERENCES signatures(operation_id),
            UNIQUE(model_id, operation_id, count)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS coll_ops (
            collective_id INTEGER PRIMARY KEY AUTOINCREMENT,
            signature_hash TEXT UNIQUE NOT NULL,
            operation_name TEXT NOT NULL,
            topology TEXT NOT NULL,
            tp_degree INTEGER NOT NULL,
            num_dims INTEGER,
            dim_param INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS model_collectives (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id INTEGER NOT NULL,
            collective_id INTEGER NOT NULL,
            count INTEGER NOT NULL DEFAULT 1,
            FOREIGN KEY (model_id) REFERENCES profiled_models(model_id),
            FOREIGN KEY (collective_id) REFERENCES coll_ops(collective_id),
            UNIQUE(model_id, collective_id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS model_collective_configs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id INTEGER NOT NULL,
            collective_id INTEGER NOT NULL,
            batch_param TEXT NOT NULL,
            model_config_dim INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (model_id) REFERENCES profiled_models(model_id),
            FOREIGN KEY (collective_id) REFERENCES coll_ops(collective_id),
            UNIQUE(model_id, collective_id)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_model_collective_configs_lookup
        ON model_collective_configs(model_id, collective_id)
    """)
    _coll_op_results(conn)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS profile_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_uid TEXT UNIQUE NOT NULL,
            device TEXT,
            gpu_name TEXT,
            engine TEXT,
            backend TEXT,
            stack_json TEXT,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


def record_run(conn, stack, started_at=None):
    """The run_id of the run ``stack`` describes (run_stack()), adding it to profile_runs when
    its run_uid is not there yet."""
    uid = stack.get("run_uid") or uuid.uuid4().hex
    row = conn.execute("SELECT run_id FROM profile_runs WHERE run_uid = ?", (uid,)).fetchone()
    if row:
        return row[0]
    cols = ["run_uid", "device", "gpu_name", "engine", "backend", "stack_json"]
    vals = [uid, stack.get("device"), stack.get("gpu_name"), stack.get("engine"), stack.get("backend"),
            json.dumps({k: v for k, v in stack.items() if k != "run_uid"}, sort_keys=True)]
    if started_at:
        cols.append("started_at")
        vals.append(started_at)
    cur = conn.execute(f"INSERT INTO profile_runs ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", vals)
    conn.commit()
    return cur.lastrowid
