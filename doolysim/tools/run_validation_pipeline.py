#!/usr/bin/env python3
"""End-to-end Dooly validation pipeline.

Runs (or re-uses) a Dooly profile DB across multiple (model, backend) pairs,
then drives `doolysim.tools.validate_predictions` over the resulting DB to
compare predicted vs vLLM-measured latencies, optionally streaming GPU clock
samples to a CSV in parallel.

Stages:
    1. (optional) Profile every (model, backend, tp) pair into the target DB.
       Wraps `profiler.run_scenario_a.py` — pairs are read from the PAIRS
       constant in that file unless --models / --backends override.
    2. (optional) Run `doolysim.tools.validate_predictions` over the DB.
    3. (optional) Plot the standard latency + inversion + chess figures via
       the matching plot_validation* tools.

This is the Python replacement for the older bash recipe at
`/home/jkim38/guidellm-out/test_scenario_a_locked_1700.sh`.

Usage examples:

    # Full pipeline at locked 1700 MHz (assumes you already locked the GPU)
    python -m doolysim.tools.run_validation_pipeline \\
        --db-path        /home/jkim38/dooly/profiler/database/dooly-locked-1700.db \\
        --workspace-dir  /home/jkim38/dooly/profiler/workspaces/locked_1700 \\
        --results-dir    /home/jkim38/dooly/results/validation_locked_1700 \\
        --models meta-llama/Llama-3.1-8B CohereLabs/c4ai-command-r7b-12-2024 \\
        --backends FLASHINFER FLASH_ATTN TRITON_ATTN

    # Skip profiling (DB already populated), just validate + plot
    python -m doolysim.tools.run_validation_pipeline ... --skip-profile

    # Validate at a denser grid, no plots
    python -m doolysim.tools.run_validation_pipeline ... \\
        --lengths 256 512 768 1024 1280 1536 1792 2048 2560 3072 \\
                  3584 4096 5120 6144 7168 8192 10240 12288 14336 16384 \\
        --no-plots
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path("/home/jkim38/dooly")
DEFAULT_PYTHON = "/home/jkim38/miniconda3/envs/dooly/bin/python"
DEFAULT_CUDA_HOME = "/usr/local/cuda-12.8"
DEFAULT_PATH_PREPEND = (
    "/home/jkim38/miniconda3/envs/dooly/bin:/usr/local/cuda-12.8/bin"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], *, cwd: Path | None = None,
         env: dict | None = None, check: bool = True,
         tail: int | None = None) -> int:
    """Run a subprocess, optionally truncating stdout to the last `tail` lines.

    Returns the exit code. Raises CalledProcessError if `check` and exit != 0.
    """
    print(f"  $ {' '.join(shlex.quote(c) for c in cmd)}")
    if tail is None:
        proc = subprocess.run(cmd, cwd=cwd, env=env, check=False)
        rc = proc.returncode
    else:
        # Stream stdout, keep only the last `tail` lines for display.
        from collections import deque
        buf: deque[str] = deque(maxlen=tail)
        proc = subprocess.Popen(
            cmd, cwd=cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            buf.append(line.rstrip("\n"))
        proc.wait()
        rc = proc.returncode
        for line in buf:
            print("    " + line)
    if check and rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)
    return rc


def _build_env() -> dict:
    """Environment with PATH/CUDA_HOME shaped for FlashInfer JIT etc."""
    env = os.environ.copy()
    env["PATH"] = f"{DEFAULT_PATH_PREPEND}:{env.get('PATH', '')}"
    env["CUDA_HOME"] = DEFAULT_CUDA_HOME
    return env


# ---------------------------------------------------------------------------
# GPU clock monitor (parallel to the profile run)
# ---------------------------------------------------------------------------

@dataclass
class ClockMonitor:
    csv_path: Path
    period_ms: int = 200
    proc: subprocess.Popen | None = field(default=None, init=False)

    def start(self) -> None:
        cmd = [
            "nvidia-smi",
            "--query-gpu=timestamp,clocks.current.sm,temperature.gpu,power.draw",
            "--format=csv,noheader",
            f"-lms", str(self.period_ms),
        ]
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        f = open(self.csv_path, "w")
        self.proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        time.sleep(0.5)
        print(f"  [clock-monitor] started, logging to {self.csv_path}")

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            self.proc.send_signal(signal.SIGTERM)
            self.proc.wait(timeout=3)
        except Exception:
            self.proc.kill()
        self.proc = None
        print(f"  [clock-monitor] stopped")

    def histogram(self, top_n: int = 10) -> str:
        if not self.csv_path.exists():
            return "(no clock log)"
        from collections import Counter
        counts: Counter = Counter()
        with open(self.csv_path) as f:
            for line in f:
                parts = line.split(",")
                if len(parts) < 2:
                    continue
                clk = parts[1].strip()
                if clk:
                    counts[clk] += 1
        lines = [f"  {n:>6d}  {clk}" for clk, n in counts.most_common(top_n)]
        return "\n".join(lines) if lines else "(empty clock log)"


# ---------------------------------------------------------------------------
# Stage 1: profile via run_scenario_a.py
# ---------------------------------------------------------------------------

def stage_profile(args: argparse.Namespace, env: dict) -> None:
    """Profile every (model, backend) pair into args.db_path.

    Skips pairs already present in the DB (run_scenario_a.py's logic). If
    args.fresh_db is set and the DB exists, refuses to proceed (to avoid
    silent clobber of prior runs).
    """
    db = args.db_path
    if db.exists():
        if args.fresh_db:
            raise FileExistsError(
                f"--fresh-db requested but {db} already exists. "
                f"Move it aside (e.g. `mv {db} {db}.bak`) before re-running."
            )
        print(f"  [profile] DB already exists at {db}; "
              f"run_scenario_a will skip already-profiled pairs.")

    cmd = [
        args.python,
        str(REPO / "profiler" / "run_scenario_a.py"),
        "--db-path", str(db),
        "--workspace-dir", str(args.workspace_dir),
        "--max-batch-size", str(args.max_batch_size),
        "--max-seq-len", str(args.max_seq_len),
        "--tp", str(args.tp),
        "--num-gpus", str(args.num_gpus),
    ]

    args.workspace_dir.mkdir(parents=True, exist_ok=True)
    db.parent.mkdir(parents=True, exist_ok=True)

    # Optional flashinfer cache wipe — recommended on first run after a
    # toolchain change so the JIT compiles cleanly.
    if args.flush_flashinfer_cache:
        cache = Path.home() / ".cache" / "flashinfer"
        if cache.exists():
            print(f"  [profile] flushing {cache}")
            shutil.rmtree(cache, ignore_errors=True)

    # Pair-source resolution is delegated to run_scenario_a.py which now
    # accepts either --config <file> or --models/--backends. We forward the
    # orchestrator's flags directly when present.
    if args.profile_config is not None:
        cmd += ["--config", str(args.profile_config)]
    elif args.models or args.backends:
        if args.models:
            cmd += ["--models", *args.models]
        if args.backends:
            cmd += ["--backends", *args.backends]

    monitor = None
    if args.clock_log is not None:
        monitor = ClockMonitor(args.clock_log)
        monitor.start()

    t0 = time.time()
    try:
        _run(cmd, cwd=REPO / "profiler", env=env, tail=40)
    finally:
        if monitor is not None:
            monitor.stop()
            print("  [profile] clock distribution during profile:")
            print(monitor.histogram())
    print(f"  [profile] elapsed: {time.time() - t0:.1f}s\n")


# ---------------------------------------------------------------------------
# Stage 2: validate_predictions
# ---------------------------------------------------------------------------

def stage_validate(args: argparse.Namespace, env: dict) -> None:
    cmd = [
        args.python, "-m", "doolysim.tools.validate_predictions",
        "--db-path", str(args.db_path),
        "--output-dir", str(args.results_dir),
        "--lengths", *(str(L) for L in args.lengths),
        "--gpu", str(args.gpu),
        "--gpu-memory", str(args.gpu_memory),
        "--max-num-seqs", str(args.max_num_seqs),
        "--warmup", str(args.warmup),
        "--iterations", str(args.iterations),
        "--buffer-tokens", str(args.buffer_tokens),
    ]
    if args.models:
        cmd += ["--models", *args.models]
    if args.backends:
        cmd += ["--backends", *args.backends]
    if args.skip_vllm:
        cmd.append("--skip-vllm")
    if args.skip_predict:
        cmd.append("--skip-predict")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    _run(cmd, cwd=REPO, env=env, tail=120)
    print(f"  [validate] elapsed: {time.time() - t0:.1f}s\n")


# ---------------------------------------------------------------------------
# Stage 3: plots
# ---------------------------------------------------------------------------

def stage_plots(args: argparse.Namespace, env: dict) -> None:
    summary = args.results_dir / "validation_summary.json"
    if not summary.exists():
        print(f"  [plots] no summary at {summary}; skipping plot stage.")
        return

    # Latency + inversion overlay
    # _run(
    #     [args.python, "-m", "doolysim.tools.plot_validation",
    #      "--summary", str(summary)],
    #     cwd=REPO, env=env, tail=4,
    # )
    # Chess heatmap
    _run(
        [args.python, "-m", "doolysim.tools.plot_validation_chess",
         "--summary", str(summary),
         "--output-dir", str(args.results_dir),
         "--workload", "prefill",
         "--source", "predicted"],
        cwd=REPO, env=env, tail=4,
    )


# ---------------------------------------------------------------------------
# Inspection helper
# ---------------------------------------------------------------------------

def print_db_contents(db: Path) -> None:
    if not db.exists():
        print(f"  (no DB at {db} yet)")
        return
    with sqlite3.connect(str(db)) as con:
        rows = con.execute(
            "SELECT model_name, backend, tp_degree FROM profiled_models "
            "ORDER BY model_name, backend"
        ).fetchall()
    if not rows:
        print(f"  (DB exists but is empty: {db})")
        return
    print(f"  Profiled (model, backend, tp) in {db}:")
    for r in rows:
        print(f"    {r[0]} | {r[1]} | tp={r[2]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Paths
    ap.add_argument("--db-path", type=Path, required=True,
                    help="SQLite DB to populate / validate against.")
    ap.add_argument("--workspace-dir", type=Path, required=True,
                    help="Workspace root for traces + per-tp profile dirs "
                         "(passed to run_scenario_a.py).")
    ap.add_argument("--results-dir", type=Path, required=True,
                    help="Directory for validate_predictions outputs and plots.")

    # Profile-stage knobs
    ap.add_argument("--max-batch-size", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=16384)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--num-gpus", type=int, default=1)
    ap.add_argument("--fresh-db", action="store_true",
                    help="Refuse to start if --db-path already exists.")
    ap.add_argument("--profile-config", type=Path, default=None,
                    help="YAML/JSON pair-list config forwarded to "
                         "run_scenario_a.py --config. When set, --models / "
                         "--backends are *not* forwarded to the profile stage "
                         "(they still apply to validation).")
    ap.add_argument("--flush-flashinfer-cache", action="store_true",
                    help="rm -rf ~/.cache/flashinfer before profiling.")

    # Filters (forwarded to validate_predictions; profile stage notes only)
    ap.add_argument("--models", nargs="*", default=None,
                    help="Restrict validation to these models.")
    ap.add_argument("--backends", nargs="*", default=None,
                    help="Restrict validation to these backends.")

    # Validate-stage knobs
    ap.add_argument("--lengths", type=int, nargs="+",
                    default=[1024, 2048, 4096, 8192, 16384],
                    help="Prefill lengths to validate at. "
                         "(default: 1K, 2K, 4K, 8K, 16K)")
    ap.add_argument("--gpu", type=str, default="0")
    ap.add_argument("--gpu-memory", type=float, default=0.85)
    ap.add_argument("--max-num-seqs", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--buffer-tokens", type=int, default=256)
    ap.add_argument("--skip-vllm", action="store_true",
                    help="Forward to validate_predictions: skip the vLLM-direct "
                         "measurements (predictor only).")
    ap.add_argument("--skip-predict", action="store_true",
                    help="Forward to validate_predictions: skip predictor "
                         "(measurements only).")

    # Stage selection
    ap.add_argument("--skip-profile", action="store_true",
                    help="Don't run run_scenario_a.py; assume DB is populated.")
    ap.add_argument("--skip-validate", action="store_true",
                    help="Don't run validate_predictions.")
    ap.add_argument("--no-plots", action="store_true",
                    help="Don't auto-generate plots after validation.")

    # Misc
    ap.add_argument("--python", default=DEFAULT_PYTHON,
                    help=f"Python interpreter to use. (default: {DEFAULT_PYTHON})")
    ap.add_argument("--clock-log", type=Path, default=None,
                    help="If set, stream nvidia-smi clock samples to this CSV "
                         "during the profile stage.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    env = _build_env()

    print("=" * 72)
    print(" Dooly validation pipeline")
    print("=" * 72)
    print(f"  DB:            {args.db_path}")
    print(f"  Workspace:     {args.workspace_dir}")
    print(f"  Results:       {args.results_dir}")
    print(f"  Lengths:       {args.lengths}")
    print(f"  TP / Batch:    tp={args.tp}, max_batch_size={args.max_batch_size}, max_seq_len={args.max_seq_len}")
    print(f"  Stages:        "
          f"{'profile, ' if not args.skip_profile else ''}"
          f"{'validate' if not args.skip_validate else 'no-validate'}"
          f"{', plots' if not args.no_plots else ''}")
    print()

    # Lock-state snapshot — useful for paper repro.
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=clocks.gr,clocks.current.sm,clocks.max.sm,power.limit",
             "--format=csv,noheader"],
            text=True,
        ).strip()
        print(f"  GPU clock state: {out}")
    except Exception as e:
        print(f"  (couldn't query clock state: {e})")
    print()

    # Stage 1
    if not args.skip_profile:
        print("--- Stage 1: Profile (run_scenario_a.py) ---")
        stage_profile(args, env)
    else:
        print("--- Stage 1: Profile  [skipped, --skip-profile] ---\n")

    print("--- DB contents ---")
    print_db_contents(args.db_path)
    print()

    # Stage 2
    if not args.skip_validate:
        print("--- Stage 2: Validate (validate_predictions.py) ---")
        stage_validate(args, env)
    else:
        print("--- Stage 2: Validate  [skipped, --skip-validate] ---\n")

    # Stage 3
    if not args.no_plots:
        print("--- Stage 3: Plots ---")
        stage_plots(args, env)
    else:
        print("--- Stage 3: Plots  [skipped, --no-plots] ---\n")

    print("Pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
