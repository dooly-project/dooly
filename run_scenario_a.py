#!/usr/bin/env python
"""Scenario A — 7-9B dense-model selection lineup x 3 attention backends.

Skips (model, backend) pairs already present in the DB. When --num-gpus > 1,
PAIRS are round-robin partitioned across GPUs and each worker runs serially
on its own GPU in parallel with the others.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path
from typing import List, Tuple

PY = "/home/jkim38/miniconda3/envs/dooly-sglang/bin/python"
WORK = Path("/home/jkim38/dooly/profiler")

DEFAULT_DB = WORK / "database" / "dooly-eval.db"
DEFAULT_WS = WORK / "workspaces" / "sglang-eval"

# Backend names use SGLang's native lowercase tokens. SGLang's ServerArgs
# rejects the legacy vLLM-style uppercase names (FLASHINFER / FLASH_ATTN /
# TRITON_ATTN) via argparse choices=ATTENTION_BACKEND_CHOICES.
PAIRS: list[Tuple[str, str]] = [
    ("meta-llama/Meta-Llama-3-8B",        "flashinfer"),
    ("meta-llama/Meta-Llama-3-8B",        "fa3"),
    ("meta-llama/Meta-Llama-3-8B",        "triton"),
    ("meta-llama/Llama-3.1-8B",           "flashinfer"),
    ("meta-llama/Llama-3.1-8B",           "fa3"),
    ("meta-llama/Llama-3.1-8B",           "triton"),
    ("meta-llama/Llama-2-7b-hf",          "flashinfer"),
    ("meta-llama/Llama-2-7b-hf",          "fa3"),
    ("meta-llama/Llama-2-7b-hf",          "triton"),
    ("Qwen/Qwen2.5-7B",                   "flashinfer"),
    ("Qwen/Qwen2.5-7B",                   "fa3"),
    ("Qwen/Qwen2.5-7B",                   "triton"),
    ("Qwen/Qwen3-8B",                     "flashinfer"),
    ("Qwen/Qwen3-8B",                     "fa3"),
    ("Qwen/Qwen3-8B",                     "triton"),
    ("CohereLabs/aya-expanse-8b",         "flashinfer"),
    ("CohereLabs/aya-expanse-8b",         "fa3"),
    ("CohereLabs/aya-expanse-8b",         "triton"),
    ("CohereLabs/c4ai-command-r7b-12-2024", "flashinfer"),
    ("CohereLabs/c4ai-command-r7b-12-2024", "fa3"),
    ("CohereLabs/c4ai-command-r7b-12-2024", "triton"),
    ("mistralai/Mistral-7B-v0.3",         "flashinfer"),
    ("mistralai/Mistral-7B-v0.3",         "fa3"),
    ("mistralai/Mistral-7B-v0.3",         "triton"),
    ('mistralai/Ministral-8B-Instruct-2410', "flashinfer"),
    ('mistralai/Ministral-8B-Instruct-2410', "fa3"),
    ('mistralai/Ministral-8B-Instruct-2410', "triton"),
    ("deepseek-ai/deepseek-llm-7b-base",  "flashinfer"),
    ("deepseek-ai/deepseek-llm-7b-base",  "fa3"),
    ("deepseek-ai/deepseek-llm-7b-base",  "triton"),
    ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "flashinfer"),
    ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "fa3"),
    ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "triton"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "flashinfer"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "fa3"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "triton"),
]


_progress_lock = threading.Lock()


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
            )
            log_progress(progress, f"[gpu{gpu}] DONE: {model} {backend} exit={ec}")
        except Exception as e:
            log_progress(
                progress,
                f"[gpu{gpu}] ERROR on {model} {backend}: {type(e).__name__}: {e}"
            )
    log_progress(progress, f"[gpu{gpu}] worker done.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Scenario A multi-model profiler driver")
    ap.add_argument("--db-path", type=Path, default=DEFAULT_DB)
    ap.add_argument("--workspace-dir", type=Path, default=DEFAULT_WS)
    ap.add_argument("--max-batch-size", type=int, default=256)
    ap.add_argument("--max-seq-len", type=int, default=32768)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--num-gpus", type=int, default=1,
                    help="How many GPUs to run across in parallel. PAIRS are "
                         "round-robin partitioned; each worker uses --gpu N.")
    ap.add_argument("--gpu-ids", type=str, default=None,
                    help="Optional comma-separated GPU indices (e.g. '0,1,2,3'). "
                         "Overrides --num-gpus. Length determines number of workers.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print which pairs would run/skip without launching.")
    args = ap.parse_args()

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
            f"Scenario A started: {len(PAIRS)} pairs across {num_workers} GPU(s) "
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
            f"Scenario A started: {len(PAIRS)} pairs on GPU {gpus[0]}, "
            f"DB={args.db_path}"
        )

    # Round-robin partition so similar-cost models (different backends of the
    # same model) spread across GPUs instead of piling on one worker.
    threads: List[threading.Thread] = []
    for i, gpu in enumerate(gpus):
        my_pairs = PAIRS[i::num_workers]
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

    log_progress(progress, "Scenario A complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
