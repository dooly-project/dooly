#!/usr/bin/env python
"""Multi-(model, backend) profiler driver.

Skips (model, backend) pairs already present in the DB. When --num-gpus > 1,
pairs are round-robin partitioned across GPUs and each worker runs serially
on its own GPU in parallel with the others.

The pair list can come from three places, in priority order:
    1. ``--config <path>`` — YAML or JSON file; see schemas below.
    2. ``--models / --backends`` — Cartesian product of two CLI lists.
    3. ``DEFAULT_PAIRS`` constant in this file (the historical Scenario A list).

Config file schemas (either form is accepted; YAML examples shown):

    # Form A — explicit list of pairs (lets you exclude specific combos):
    pairs:
      - [meta-llama/Llama-3.1-8B,            FLASHINFER]
      - [meta-llama/Llama-3.1-8B,            FLASH_ATTN]
      - [CohereLabs/c4ai-command-r7b-12-2024, FLASHINFER]

    # Form B — cartesian product (concise for "every model × every backend"):
    models:
      - meta-llama/Llama-3.1-8B
      - CohereLabs/c4ai-command-r7b-12-2024
    backends:
      - FLASHINFER
      - FLASH_ATTN
      - TRITON_ATTN

Both forms accept either a list of two-element [model, backend] sequences
or a list of mappings ``{model: ..., backend: ...}``. JSON works the same.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path
from typing import List, Sequence, Tuple

PY = "/home/ubuntu/miniconda3/envs/dooly/bin/python"
WORK = Path("/home/ubuntu/dooly/profiler")

DEFAULT_DB = WORK / "database" / "dooly-eval.db"
DEFAULT_WS = WORK / "workspaces" / "scenario_a"

DEFAULT_PAIRS: list[Tuple[str, str]] = [
    # ("meta-llama/Meta-Llama-3-8B",        "FLASHINFER"),
    # ("meta-llama/Meta-Llama-3-8B",        "FLASH_ATTN"),
    # ("meta-llama/Meta-Llama-3-8B",        "TRITON_ATTN"),
    ("meta-llama/Llama-3.1-8B",           "FLASHINFER"),
    # ("meta-llama/Llama-3.1-8B",           "FLASH_ATTN"),
    # ("meta-llama/Llama-3.1-8B",           "TRITON_ATTN"),
    # ("meta-llama/Llama-2-7b-hf",          "FLASHINFER"),
    # ("meta-llama/Llama-2-7b-hf",          "FLASH_ATTN"),
    # ("meta-llama/Llama-2-7b-hf",          "TRITON_ATTN"),
    # ("Qwen/Qwen2.5-7B",                   "FLASHINFER"),
    # ("Qwen/Qwen2.5-7B",                   "FLASH_ATTN"),
    # ("Qwen/Qwen2.5-7B",                   "TRITON_ATTN"),
    # ("Qwen/Qwen3-8B",                     "FLASHINFER"),
    # ("Qwen/Qwen3-8B",                     "FLASH_ATTN"),
    # ("Qwen/Qwen3-8B",                     "TRITON_ATTN"),
    # ("CohereLabs/aya-expanse-8b",         "FLASHINFER"),
    # ("CohereLabs/aya-expanse-8b",         "FLASH_ATTN"),
    # ("CohereLabs/aya-expanse-8b",         "TRITON_ATTN"),
    ("CohereLabs/c4ai-command-r7b-12-2024", "FLASHINFER"),
    # ("CohereLabs/c4ai-command-r7b-12-2024", "FLASH_ATTN"),
    # ("CohereLabs/c4ai-command-r7b-12-2024", "TRITON_ATTN"),
    # ("mistralai/Mistral-7B-v0.3",         "FLASHINFER"),
    # ("mistralai/Mistral-7B-v0.3",         "FLASH_ATTN"),
    # ("mistralai/Mistral-7B-v0.3",         "TRITON_ATTN"),
    # ('mistralai/Ministral-8B-Instruct-2410', "FLASHINFER"),
    # ('mistralai/Ministral-8B-Instruct-2410', "FLASH_ATTN"),
    # ('mistralai/Ministral-8B-Instruct-2410', "TRITON_ATTN"),
    # ("deepseek-ai/deepseek-llm-7b-base",  "FLASHINFER"),
    # ("deepseek-ai/deepseek-llm-7b-base",  "FLASH_ATTN"),
    # ("deepseek-ai/deepseek-llm-7b-base",  "TRITON_ATTN"),
    # ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "FLASHINFER"),
    # ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "FLASH_ATTN"),
    # ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "TRITON_ATTN"),
    # ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "FLASHINFER"),
    # ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "FLASH_ATTN"),
    # ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "TRITON_ATTN"),
]


_progress_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Pair resolution (config file / CLI flags / hardcoded default)
# ---------------------------------------------------------------------------

def _load_yaml_or_json(path: Path) -> dict:
    """Read a config file as YAML if PyYAML is available and the extension
    matches, else as JSON. JSON is also accepted by YAML's loader so .yaml
    files containing pure JSON also work."""
    text = path.read_text()
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                f"Reading {path}: install PyYAML (`pip install pyyaml`) "
                f"to use .yaml configs, or convert the file to .json."
            ) from e
        return yaml.safe_load(text) or {}
    if suffix == ".json":
        return json.loads(text)
    # Fallback: try YAML if available, then JSON.
    try:
        import yaml  # type: ignore
        loaded = yaml.safe_load(text)
        if isinstance(loaded, dict):
            return loaded
    except Exception:
        pass
    return json.loads(text)


def _coerce_pair(item, source: str) -> Tuple[str, str]:
    """Normalize one entry of a `pairs:` list into (model, backend)."""
    if isinstance(item, (list, tuple)) and len(item) == 2:
        return (str(item[0]), str(item[1]))
    if isinstance(item, dict) and "model" in item and "backend" in item:
        return (str(item["model"]), str(item["backend"]))
    raise ValueError(
        f"{source}: expected a 2-element [model, backend] list or "
        f"a {{model, backend}} mapping; got {item!r}"
    )


def load_pairs_from_config(path: Path) -> List[Tuple[str, str]]:
    """Read a YAML or JSON config and return the normalized pair list.

    Accepts either schema:

        pairs:
          - [model, backend]            # or {model: ..., backend: ...}
          - ...

        # OR — cartesian-product form:
        models:    [m1, m2, ...]
        backends:  [b1, b2, ...]

    A config that contains both forms uses ``pairs`` (the explicit list).
    """
    data = _load_yaml_or_json(path)
    if not isinstance(data, dict):
        raise ValueError(
            f"{path}: top-level must be a mapping/object, got {type(data).__name__}"
        )

    if "pairs" in data:
        items = data["pairs"]
        if not isinstance(items, list):
            raise ValueError(f"{path}: 'pairs' must be a list, got {type(items).__name__}")
        return [_coerce_pair(it, f"{path} -> pairs[{i}]") for i, it in enumerate(items)]

    if "models" in data and "backends" in data:
        models = data["models"]
        backends = data["backends"]
        if not isinstance(models, list) or not isinstance(backends, list):
            raise ValueError(f"{path}: 'models' and 'backends' must each be a list.")
        return [(str(m), str(b)) for m in models for b in backends]

    raise ValueError(
        f"{path}: must contain either a 'pairs:' list or both "
        f"'models:' and 'backends:' lists. Top-level keys present: "
        f"{sorted(data.keys())}"
    )


# Config-driven knobs. Each tuple is:
#   (arg_attr, config_section, builtin_default, type_cast)
# The merge order is: CLI flag (if not None) > config value > builtin default.
#
# `config_section` is the nested section name in YAML/JSON (e.g. ``paths``).
# Flat top-level keys with the same name are also accepted as a fallback.
_CONFIG_SCHEMA: list[tuple[str, str, object, callable]] = [
    # paths
    ("db_path",        "paths",  None,                  Path),
    ("workspace_dir",  "paths",  None,                  Path),
    # sweep space
    ("max_batch_size", "sweep",  4,                     int),
    ("max_seq_len",    "sweep",  16384,                 int),
    ("tp",             "sweep",  1,                     int),
    # gpus
    ("num_gpus",       "gpus",   1,                     int),
    ("gpu_ids",        "gpus",   None,                  str),
]


def _config_get(data: dict, section: str, key: str):
    """Look up `key` in data[section] (preferred) or at top-level (fallback).
    Returns sentinel `None` if not present in either place."""
    sect = data.get(section)
    if isinstance(sect, dict) and key in sect:
        return sect[key]
    if key in data:
        return data[key]
    return None


def _merge_config_into_args(args: argparse.Namespace, config_data: dict) -> dict:
    """Apply config defaults to `args` for every attr in _CONFIG_SCHEMA.

    Returns a dict of the resolved values (for printing). Mutates `args` so
    downstream code can keep reading args.<name> directly.
    """
    resolved: dict = {}
    for attr, section, builtin_default, cast in _CONFIG_SCHEMA:
        cli_val = getattr(args, attr, None)
        cfg_val = _config_get(config_data, section, attr) if config_data else None

        # Apply DEFAULT_DB / DEFAULT_WS for paths only when nothing else supplies them.
        if attr == "db_path" and builtin_default is None:
            builtin_default = DEFAULT_DB
        if attr == "workspace_dir" and builtin_default is None:
            builtin_default = DEFAULT_WS

        if cli_val is not None:
            value = cli_val
        elif cfg_val is not None:
            value = cast(cfg_val) if cast is not None else cfg_val
        else:
            value = builtin_default

        setattr(args, attr, value)
        resolved[attr] = value
    return resolved


def resolve_pairs(args: argparse.Namespace) -> List[Tuple[str, str]]:
    """Resolve the pair list AND merge config knobs into ``args``.

    Priority for the pair list: ``--config`` > ``--models``/``--backends`` >
    ``DEFAULT_PAIRS``. Priority for every other knob in ``_CONFIG_SCHEMA``
    (paths, sweep space, gpus): CLI flag (if not None) > config value (nested
    or flat) > builtin default.
    """
    config_data: dict = {}
    pairs: List[Tuple[str, str]] | None = None

    if args.config is not None:
        config_data = _load_yaml_or_json(args.config)
        if not isinstance(config_data, dict):
            raise ValueError(
                f"{args.config}: top-level must be a mapping/object, "
                f"got {type(config_data).__name__}"
            )
        if "pairs" in config_data:
            pairs = [
                _coerce_pair(it, f"{args.config} -> pairs[{i}]")
                for i, it in enumerate(config_data["pairs"])
            ]
        elif "models" in config_data and "backends" in config_data:
            pairs = [(str(m), str(b))
                     for m in config_data["models"]
                     for b in config_data["backends"]]
        # else: pairs stay None — config might be config-only, fall through.

    if pairs is None and (args.models or args.backends):
        if not (args.models and args.backends):
            raise SystemExit(
                "[ERROR] --models and --backends must both be provided when "
                "specifying pairs via CLI flags. Pass --config <file> instead "
                "to use an explicit list."
            )
        pairs = [(m, b) for m in args.models for b in args.backends]
        print(f"[run_scenario_a] {len(pairs)} pairs from --models × --backends "
              f"(cartesian product)")
    elif pairs is not None:
        print(f"[run_scenario_a] loaded {len(pairs)} (model, backend) pairs "
              f"from {args.config}")
    else:
        pairs = list(DEFAULT_PAIRS)
        print(f"[run_scenario_a] {len(pairs)} pairs from hardcoded DEFAULT_PAIRS")

    # Merge every other config-managed knob into args.
    resolved = _merge_config_into_args(args, config_data)
    print(
        "[run_scenario_a] resolved knobs: "
        f"db_path={resolved['db_path']}, workspace_dir={resolved['workspace_dir']}, "
        f"max_batch_size={resolved['max_batch_size']}, "
        f"max_seq_len={resolved['max_seq_len']}, tp={resolved['tp']}, "
        f"num_gpus={resolved['num_gpus']}, gpu_ids={resolved['gpu_ids']!r}"
        + (f"  (config: {args.config})" if config_data else "")
    )
    return pairs


def already_profiled(db: Path, model: str, backend: str, tp: int) -> bool:
    """Check DB for existing (model, backend, tp) row.

    Uses a long busy-timeout so concurrent workers wait on the SQLite lock
    instead of raising, and catches OperationalError defensively: on a transient
    DB error we return False (i.e. "assume not profiled"). The worst case is
    re-profiling a pair, which is fine — `profiled_models` has a UNIQUE
    constraint that will reject the duplicate on insert.
    """
    if not db.exists():
        return False
    try:
        with sqlite3.connect(str(db), timeout=60.0) as con:
            (n,) = con.execute(
                "SELECT COUNT(*) FROM profiled_models "
                "WHERE model_name=? AND backend=? AND tp_degree=?",
                (model, backend, tp),
            ).fetchone()
        return n > 0
    except sqlite3.OperationalError as e:
        print(f"[WARN] already_profiled check failed for "
              f"{model}/{backend}/tp{tp}: {e}", flush=True)
        return False


def stamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_progress(progress_path: Path, msg: str) -> None:
    line = f"[{stamp()}] {msg}"
    with _progress_lock:
        print(line, flush=True)
        with open(progress_path, "a") as f:
            f.write(line + "\n")


def shard_db_path(base_db: Path, gpu: int) -> Path:
    """Return `{stem}_gpu{N}{suffix}` alongside the base DB path."""
    return base_db.with_name(f"{base_db.stem}_gpu{gpu}{base_db.suffix}")


def run_one(
    model: str,
    backend: str,
    db: Path,
    ws: Path,
    log_dir: Path,
    max_batch_size: int,
    max_seq_len: int,
    tp: int,
    gpu: int,
    quantization: str | None = None,
) -> int:
    safe = model.replace("/", "_")
    log_path = log_dir / f"{safe}__{backend}__gpu{gpu}.log"

    cmd = [
        PY, "-m", "dooly.run.run-e2e",
        "--models", model,
        "--attention-backend", backend,
        "--max-batch-size", str(max_batch_size),
        "--max-seq-len", str(max_seq_len),
        "--tp", str(tp),
        "--workspace-dir", str(ws),
        "--db-path", str(db),
        "--gpu", str(gpu),
    ]
    if quantization:
        cmd.extend(["--quantization", quantization])
    env = os.environ.copy()
    env["DOOLY_TIMING_SHARD"] = f"gpu{gpu}"
    with open(log_path, "w") as lf:
        return subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT,
                               cwd=str(WORK), env=env)


def worker_loop(
    gpu: int,
    my_pairs: List[Tuple[str, str]],
    args: argparse.Namespace,
    log_dir: Path,
    progress: Path,
    num_workers: int,
) -> None:
    # Each worker writes to its own per-GPU DB to avoid SQLite lock contention.
    # With num_workers == 1 we preserve the legacy behavior of writing directly
    # to the base DB (no _gpu{N} suffix).
    worker_db = args.db_path if num_workers == 1 else shard_db_path(args.db_path, gpu)
    log_progress(
        progress,
        f"[gpu{gpu}] worker started: {len(my_pairs)} pairs assigned, DB={worker_db}"
    )
    for model, backend in my_pairs:
        try:
            if already_profiled(worker_db, model, backend, args.tp):
                log_progress(progress, f"[gpu{gpu}] SKIP (already in DB): {model} {backend}")
                continue

            if args.dry_run:
                log_progress(progress, f"[gpu{gpu}] DRY-RUN would start: {model} {backend}")
                continue

            log_progress(progress, f"[gpu{gpu}] START: {model} {backend}")
            ec = run_one(
                model=model, backend=backend,
                db=worker_db, ws=args.workspace_dir, log_dir=log_dir,
                max_batch_size=args.max_batch_size,
                max_seq_len=args.max_seq_len,
                tp=args.tp,
                gpu=gpu,
                quantization=args.quantization,
            )
            log_progress(progress, f"[gpu{gpu}] DONE: {model} {backend} exit={ec}")
        except Exception as e:
            log_progress(
                progress,
                f"[gpu{gpu}] ERROR on {model} {backend}: {type(e).__name__}: {e}"
            )
    log_progress(progress, f"[gpu{gpu}] worker done.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Multi-(model, backend) profiler driver.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # CLI defaults are intentionally None for any flag that can be supplied
    # by --config; the merge logic in resolve_pairs() applies builtin
    # defaults when neither CLI nor config sets a value.
    ap.add_argument("--db-path", type=Path, default=None)
    ap.add_argument("--workspace-dir", type=Path, default=None)
    ap.add_argument("--max-batch-size", type=int, default=None)
    ap.add_argument("--max-seq-len", type=int, default=None)
    ap.add_argument("--tp", type=int, default=None)
    ap.add_argument("--num-gpus", type=int, default=None,
                    help="How many GPUs to run across in parallel. Pairs are "
                         "round-robin partitioned; each worker uses --gpu N. "
                         "(default: 1, or whatever the config supplies)")
    ap.add_argument("--gpu-ids", type=str, default=None,
                    help="Optional comma-separated GPU indices (e.g. '0,1,2,3'). "
                         "Overrides --num-gpus. Length determines number of workers.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print which pairs would run/skip without launching.")
    ap.add_argument("--quantization", type=str, default=None,
                    choices=["fp8", "awq", "gptq"],
                    help="Optional quantization scheme. 'fp8' requires Hopper+ (H100). "
                         "Combined with load_format=dummy, vLLM initializes random "
                         "quantized weights — no quantized checkpoint needed.")

    # ---- Pair source (priority: --config > --models/--backends > DEFAULT_PAIRS) ----
    ap.add_argument("--config", type=Path, default=None,
                    help="YAML or JSON config that supplies the (model, backend) "
                         "pairs to profile. See docstring for accepted schemas.")
    ap.add_argument("--models", nargs="*", default=None,
                    help="List of models to profile (used with --backends to "
                         "form a cartesian product). Ignored when --config is set.")
    ap.add_argument("--backends", nargs="*", default=None,
                    help="List of attention backends. See --models.")
    ap.add_argument("--print-pairs", action="store_true",
                    help="Print the resolved pair list and exit (no profiling).")
    args = ap.parse_args()

    pairs = resolve_pairs(args)
    if args.print_pairs:
        for m, b in pairs:
            print(f"  {m} | {b}")
        return 0
    if not pairs:
        print("[ERROR] empty pair list. Pass --config <file> or "
              "--models/--backends, or populate DEFAULT_PAIRS in source.")
        return 2

    if args.gpu_ids:
        gpus = [int(x) for x in args.gpu_ids.split(",") if x.strip() != ""]
    else:
        if args.num_gpus < 1:
            print(f"[ERROR] --num-gpus must be >= 1 (got {args.num_gpus})")
            return 2
        gpus = list(range(args.num_gpus))
    num_workers = len(gpus)

    ws: Path = args.workspace_dir
    log_dir = ws / "logs"
    progress = ws / "progress.log"
    log_dir.mkdir(parents=True, exist_ok=True)

    if num_workers > 1:
        shard_paths = [shard_db_path(args.db_path, g) for g in gpus]
        log_progress(
            progress,
            f"Profile run started: {len(pairs)} pairs across {num_workers} GPU(s) "
            f"{gpus}. Each worker writes its own DB; merge after with:"
        )
        log_progress(
            progress,
            "  python merge_dbs.py --input " + " ".join(str(p) for p in shard_paths)
            + f" --output {args.db_path}"
        )
    else:
        log_progress(
            progress,
            f"Profile run started: {len(pairs)} pairs on GPU {gpus[0]}, "
            f"DB={args.db_path}"
        )

    # Round-robin partition so similar-cost models (different backends of the
    # same model) spread across GPUs instead of piling on one worker.
    threads: List[threading.Thread] = []
    for i, gpu in enumerate(gpus):
        my_pairs = pairs[i::num_workers]
        t = threading.Thread(
            target=worker_loop,
            args=(gpu, my_pairs, args, log_dir, progress, num_workers),
            name=f"gpu{gpu}",
            daemon=False,
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    log_progress(progress, "Profile run complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
