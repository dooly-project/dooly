"""Validate Dooly predictions against vLLM-direct measurements.

For every (model, backend, tp) profiled in a Dooly DB, runs:
  1. db_predictor against each scenario to get the predicted latency.
  2. measure_vllm_direct under the *current* GPU clock state to get the
     measured latency.
Then prints a side-by-side comparison and saves all results to JSON.

Workload modes (mirrors `doolysim.benchmark.run_comparison`):
  - prefill : single prefill of length L  (--lengths picks L; default flow)
  - decode  : single decode step at KV cache size K  (--kv-sizes picks K)
  - both    : sweep both axes back-to-back

Typical usage (after `sudo nvidia-smi -lgc 1700` if you want locked clocks):

    # Prefill-only (unchanged from before):
    python -m doolysim.tools.validate_predictions \\
        --db-path  /home/jkim38/dooly/profiler/database/clean/dooly-scenario-a-locked-1700.db \\
        --output-dir /home/jkim38/dooly/results/validation_locked_1700 \\
        --lengths 1024 2048 4096 8192 16384

    # Decode-only:
    python -m doolysim.tools.validate_predictions \\
        --db-path .../dooly.db --output-dir .../validation_decode \\
        --workload decode --kv-sizes 1024 2048 4096 8192 16384

    # Both axes in one run:
    python -m doolysim.tools.validate_predictions \\
        --db-path .../dooly.db --output-dir .../validation_both \\
        --workload both \\
        --lengths  512 1024 2048 4096 8192 16384 \\
        --kv-sizes 1024 2048 4096 8192 16384

Optional filters:
    --models  meta-llama/Llama-3.1-8B [...]
    --backends FLASHINFER FLASH_ATTN TRITON_ATTN

Other useful flags:
    --skip-vllm     only run the predictor (no vLLM measurement)
    --skip-predict  only run vLLM measurements (no predictor)
    --gpu-memory 0.85
    --max-num-seqs 8
    --warmup 2 --iterations 3
    --buffer-tokens 256   (added to axis size for vLLM's max_model_len)
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent  # /home/jkim38/dooly
DEFAULT_OUTPUT = REPO / "results" / "validation"

# Where to find db_predictor.py — defaults to the user's working location.
# Override with --db-predictor if it lives elsewhere.
DEFAULT_DB_PREDICTOR = "/home/jkim38/guidellm-out/db_predictor.py"


@dataclass
class Combo:
    model: str
    backend: str
    tp: int
    dtype: str = "bfloat16"
    kv_cache_dtype: str = "auto"

    @property
    def safe_model(self) -> str:
        return self.model.replace("/", "__")

    def __str__(self) -> str:
        return f"{self.model} | {self.backend} | tp={self.tp}"


@dataclass
class Scenario:
    """One evaluation point: a workload type and a single axis value."""
    workload: str       # 'prefill' or 'decode'
    axis_value: int     # prefill length OR decode kv_cache_size

    @property
    def tag(self) -> str:
        # File-name-safe label, e.g. 'prefill_4096' or 'decode_kv_8192'.
        if self.workload == "prefill":
            return f"prefill_{self.axis_value}"
        return f"decode_kv_{self.axis_value}"

    def axis_label(self) -> str:
        return "L" if self.workload == "prefill" else "KV"

    def __str__(self) -> str:
        return f"{self.workload}={self.axis_value}"


def build_scenarios(workloads: list[str], lengths: list[int],
                    kv_sizes: list[int]) -> list[Scenario]:
    """Construct the scenario list driven by --workload / --lengths / --kv-sizes."""
    out: list[Scenario] = []
    if "prefill" in workloads:
        out.extend(Scenario("prefill", L) for L in lengths)
    if "decode" in workloads:
        out.extend(Scenario("decode", K) for K in kv_sizes)
    return out


def discover_combos(db_path: str) -> list[Combo]:
    """Read profiled_models from the DB and return every deployment
    (model, backend, tp, dtype, kv_cache_dtype)."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DB not found: {db_path}")
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT model_name, backend, tp_degree, dtype, kv_cache_dtype FROM profiled_models "
            "ORDER BY model_name, backend, tp_degree, dtype, kv_cache_dtype"
        ).fetchall()
    return [Combo(model=m, backend=b, tp=t, dtype=d, kv_cache_dtype=k) for m, b, t, d, k in rows]


def write_batch_config(path: Path, combo: Combo, scenario: Scenario) -> None:
    """Write a batch config compatible with db_predictor and measure_vllm_direct.

    For prefill scenarios:
        prefill_requests = [{tokens: L, kv_cache: 0}], decode_requests = []
    For decode scenarios (mirroring run_comparison.py's "single decode step
    at KV=K" definition), we set `default_output_len: 2` so vLLM runs
    exactly one decode step at KV=K (the prefill produces token 1 itself,
    so max_tokens=2 produces 1 prefill + 1 decode in the trace).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg: dict = {
        "model": combo.model,
        "backend": combo.backend,
        "tensor_parallel_size": combo.tp,
        "dtype": combo.dtype,
        "kv_cache_dtype": combo.kv_cache_dtype,
    }
    if scenario.workload == "prefill":
        cfg["description"] = f"{combo.model} | {combo.backend} | prefill={scenario.axis_value}"
        cfg["prefill_requests"] = [{"tokens": scenario.axis_value, "kv_cache": 0}]
        cfg["decode_requests"] = []
        # output_len defaults to 1 for prefill_requests in measure_vllm_direct
        # (1 prefill, 0 decode) — no need to set it explicitly.
    elif scenario.workload == "decode":
        cfg["description"] = (
            f"{combo.model} | {combo.backend} | decode kv_cache={scenario.axis_value}"
        )
        cfg["prefill_requests"] = []
        cfg["decode_requests"] = [{"tokens": 1, "kv_cache": scenario.axis_value}]
        # Force exactly one decode step at the configured KV. With
        # default_output_len=2, vLLM does: 1 prefill (builds KV to K) + 1
        # decode (at KV=K). The trace yields a single decode event at the
        # exact KV the user configured.
        cfg["default_output_len"] = 2
    else:
        raise ValueError(f"Unknown workload: {scenario.workload!r}")
    path.write_text(json.dumps(cfg, indent=2))


def run_predictor(db_path: str, batch_cfg: Path, out_path: Path,
                  predictor_script: str, python_exe: str) -> dict | None:
    """Invoke db_predictor.py and load the resulting JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_exe, predictor_script,
        "--db-path", db_path,
        "--batch-config", str(batch_cfg),
        "--json-output", str(out_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  [predict] FAILED: {res.stderr.strip().splitlines()[-1] if res.stderr else 'no stderr'}")
        return None
    if not out_path.exists():
        return None
    return json.loads(out_path.read_text())


def run_vllm_direct(combo: Combo, batch_cfg: Path, out_dir: Path,
                    out_filename: str, python_exe: str,
                    max_model_len: int, max_num_seqs: int,
                    gpu_memory: float, warmup: int, iterations: int,
                    cuda_visible_devices: str,
                    quantization: str | None = None) -> dict | None:
    """Invoke measure_vllm_direct (one process = one vLLM cold start) and
    load the resulting JSON.

    `iterations` is the *inner* count passed to measure_vllm_direct. When
    you want N independent cold-start samples, call this function N times
    with iterations=1 and a unique `out_filename` each time."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / out_filename
    if out_file.exists():
        return json.loads(out_file.read_text())
    cmd = [
        python_exe, "-m", "doolysim.benchmark.measure_vllm_direct",
        "--batch-config", str(batch_cfg),
        "--tensor-parallel-size", str(combo.tp),
        "--attention-backend", combo.backend,
        "--max-model-len", str(max_model_len),
        "--max-num-seqs", str(max_num_seqs),
        "--gpu-memory", str(gpu_memory),
        "--warmup", str(warmup),
        "--iterations", str(iterations),
        "--out-dir", str(out_dir),
        "--output", out_filename,
    ]
    if quantization:
        cmd.extend(["--quantization", quantization])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(REPO))
    if res.returncode != 0:
        last = res.stderr.strip().splitlines()[-1] if res.stderr else "no stderr"
        print(f"  [vllm] FAILED: {last}")
        return None
    if not out_file.exists():
        return None
    return json.loads(out_file.read_text())


def _measured_value(vllm_data: dict, workload: str) -> float | None:
    """Extract the right `measured` latency for this workload from a
    measure_vllm_direct result dict.

    For decode rows, prefer `first_decode_latency_ms` — the very first
    decode step in the trace, which has KV cache = exactly the configured
    prompt length (no KV growth). This is the value that matches what
    db_predictor predicts. Falls back to `total_latency_ms` if the new
    field isn't present (older measure_vllm_direct output).

    For prefill rows, use `prefill_latency_ms` (the trace-classified
    median across prefill steps) when available, else `total_latency_ms`.
    """
    res = vllm_data.get("results", {})
    if workload == "decode":
        v = res.get("first_decode_latency_ms")
        if v is not None:
            return v
    elif workload == "prefill":
        v = res.get("prefill_latency_ms")
        if v is not None:
            return v
    return res.get("total_latency_ms")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db-path", required=True,
                    help="Dooly profiled DB to validate.")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT),
                    help="Where to write predictor outputs, vLLM measurements, and the summary table.")
    ap.add_argument("--workload", choices=["prefill", "decode", "both"],
                    default="prefill",
                    help="Which workload axes to sweep. 'prefill' validates "
                         "single-prefill batches at each --lengths value; "
                         "'decode' validates single-decode-step batches at "
                         "each --kv-sizes value; 'both' does the union. "
                         "(default: prefill)")
    ap.add_argument("--lengths", type=int, nargs="+",
                    default=[512, 1024, 2048, 4096, 8192, 16384],
                    help="Prefill sizes to validate (used when "
                         "--workload prefill or both).")
    ap.add_argument("--kv-sizes", type=int, nargs="+",
                    default=[512, 1024, 2048, 4096, 8192, 16384],
                    help="Decode KV-cache sizes to validate (used when "
                         "--workload decode or both). Each entry sweeps a "
                         "single decode step at that pre-existing KV size.")
    ap.add_argument("--models", nargs="*", default=None,
                    help="Restrict to these model names (default: all in DB).")
    ap.add_argument("--backends", nargs="*", default=None,
                    help="Restrict to these backends (default: all in DB).")
    ap.add_argument("--skip-vllm", action="store_true",
                    help="Skip vLLM measurements (predictions only).")
    ap.add_argument("--skip-predict", action="store_true",
                    help="Skip predictor (vLLM measurements only).")
    ap.add_argument("--db-predictor", default=DEFAULT_DB_PREDICTOR,
                    help=f"Path to db_predictor.py (default: {DEFAULT_DB_PREDICTOR}).")
    ap.add_argument("--python-exe", default=sys.executable,
                    help="Python executable to use for subprocesses (defaults to the current interpreter).")
    ap.add_argument("--gpu", type=str, default="0",
                    help="GPU index for CUDA_VISIBLE_DEVICES.")
    ap.add_argument("--gpu-memory", type=float, default=0.85)
    ap.add_argument("--max-num-seqs", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=2,
                    help="Warmup iterations inside each measure_vllm_direct "
                         "process (warmup is not profiled). (default: 2)")
    ap.add_argument("--iterations", type=int, default=1,
                    help="Inner timed iterations PER measure_vllm_direct call. "
                         "Default 1 so each call captures exactly one "
                         "vLLM-cold-start sample of (prefill + decode steps). "
                         "Use --num-runs for the outer loop.")
    ap.add_argument("--num-runs", type=int, default=3,
                    help="Number of independent measure_vllm_direct invocations "
                         "per (combo, scenario). Each invocation reloads vLLM "
                         "from scratch, so all samples are cold-start "
                         "independent. The summary uses the median across "
                         "samples; each raw sample is also stored. (default: 3)")
    ap.add_argument("--buffer-tokens", type=int, default=256,
                    help="Added to each prefill length for max_model_len (vLLM needs prompt + at_least_1_output).")
    ap.add_argument("--quantization", type=str, default=None, choices=["fp8", "awq", "gptq"],
                    help="Optional quantization scheme passed through to vLLM. 'fp8' requires Hopper+.")
    args = ap.parse_args()

    db_path = os.path.abspath(args.db_path)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configs_dir = output_dir / "batch_configs"
    pred_dir = output_dir / "dooly_predictions"
    vllm_dir = output_dir / "vllm_measurements"

    combos = discover_combos(db_path)
    if args.models:
        combos = [c for c in combos if c.model in args.models]
    if args.backends:
        combos = [c for c in combos if c.backend in args.backends]
    if not combos:
        print("No (model, backend, tp) combinations match the filters.")
        return 1

    workloads = ["prefill", "decode"] if args.workload == "both" else [args.workload]
    scenarios = build_scenarios(workloads, args.lengths, args.kv_sizes)
    if not scenarios:
        print(f"No scenarios to run. Check --workload / --lengths / --kv-sizes.")
        return 1

    print(f"DB: {db_path}")
    print(f"Output dir: {output_dir}")
    print(f"Combinations to validate: {len(combos)}")
    for c in combos:
        print(f"  - {c}")
    print(f"Workloads: {workloads}")
    if "prefill" in workloads:
        print(f"  prefill lengths: {args.lengths}")
    if "decode" in workloads:
        print(f"  decode KV sizes: {args.kv_sizes}")
    print(f"Total scenarios per (model × backend): {len(scenarios)}")
    print()

    # Make sure ninja and nvcc are reachable for FlashInfer JIT inside vLLM.
    env_path = os.environ.get("PATH", "")
    py_bindir = str(Path(args.python_exe).parent)
    if py_bindir not in env_path.split(":"):
        os.environ["PATH"] = f"{py_bindir}:{env_path}"
    if "/usr/local/cuda-12.8/bin" not in os.environ["PATH"]:
        os.environ["PATH"] = f"/usr/local/cuda-12.8/bin:{os.environ['PATH']}"
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-12.8")

    results: list[dict] = []

    t0 = time.time()
    for combo in combos:
        for sc in scenarios:
            cfg_path = configs_dir / (
                f"{combo.safe_model}__{combo.backend}__{sc.tag}.json"
            )
            write_batch_config(cfg_path, combo, sc)

            print(f"[{combo.safe_model} | {combo.backend} | {sc.axis_label()}={sc.axis_value}]")

            pred_data = None
            if not args.skip_predict:
                pred_path = pred_dir / (
                    f"pred_{combo.safe_model}__{combo.backend}__{sc.tag}.json"
                )
                pred_data = run_predictor(db_path, cfg_path, pred_path,
                                          args.db_predictor, args.python_exe)
                if pred_data is not None:
                    print(f"  predicted: {pred_data['results']['total_latency_ms']:.2f} ms")

            # Outer loop: --num-runs independent measure_vllm_direct invocations
            # (each is a separate vLLM cold start, so samples are independent).
            measured_samples: list[float] = []
            sample_dicts: list[dict] = []   # full per-run dicts for diagnostics
            if not args.skip_vllm:
                mml = sc.axis_value + args.buffer_tokens
                for run_idx in range(args.num_runs):
                    vllm_filename = (
                        f"vllm_{combo.backend}_{sc.tag}_run{run_idx}.json"
                    )
                    vllm_data = run_vllm_direct(
                        combo, cfg_path, vllm_dir / combo.safe_model,
                        vllm_filename, args.python_exe,
                        max_model_len=mml,
                        max_num_seqs=args.max_num_seqs,
                        gpu_memory=args.gpu_memory,
                        warmup=args.warmup, iterations=args.iterations,
                        cuda_visible_devices=args.gpu,
                        quantization=args.quantization,
                    )
                    if vllm_data is None:
                        print(f"  [run {run_idx + 1}/{args.num_runs}] FAILED")
                        continue
                    v = _measured_value(vllm_data, sc.workload)
                    if v is None:
                        print(f"  [run {run_idx + 1}/{args.num_runs}] no latency in result")
                        continue
                    measured_samples.append(v)
                    sample_dicts.append({
                        "run": run_idx,
                        "total_latency_ms":      vllm_data["results"].get("total_latency_ms"),
                        "first_decode_latency_ms": vllm_data["results"].get("first_decode_latency_ms"),
                        "prefill_latency_ms":    vllm_data["results"].get("prefill_latency_ms"),
                        "decode_latency_ms":     vllm_data["results"].get("decode_latency_ms"),
                        "num_prefill_steps":     vllm_data["results"].get("num_prefill_steps"),
                        "num_decode_steps":      vllm_data["results"].get("num_decode_steps"),
                    })
                    print(f"  [run {run_idx + 1}/{args.num_runs}] measured ({sc.workload}): {v:.2f} ms")

            row = {
                "model": combo.model,
                "backend": combo.backend,
                "tp": combo.tp,
                "workload": sc.workload,
                # `length` is preserved as the universal axis field for backward
                # compatibility with downstream tools (plot_validation,
                # plot_validation_chess) — for decode rows it carries the
                # KV-cache size. `kv_cache_size` is also written for clarity.
                "length": sc.axis_value,
            }
            if sc.workload == "decode":
                row["kv_cache_size"] = sc.axis_value
            if pred_data is not None:
                row["predicted_ms"] = pred_data["results"]["total_latency_ms"]
            if measured_samples:
                # Headline `measured_ms` = median across the cold-start samples.
                # Per-run raw values are preserved in `measured_samples_ms` (and
                # the per-run latency breakdowns in `samples`) for downstream
                # plots / stats.
                import numpy as np  # local import to keep top of file lean
                row["measured_ms"]         = float(np.median(measured_samples))
                row["measured_mean_ms"]    = float(np.mean(measured_samples))
                row["measured_min_ms"]     = float(np.min(measured_samples))
                row["measured_max_ms"]     = float(np.max(measured_samples))
                row["measured_std_ms"]     = float(np.std(measured_samples))
                row["measured_samples_ms"] = measured_samples
                row["samples"]             = sample_dicts
            if "predicted_ms" in row and "measured_ms" in row:
                d = row["predicted_ms"] - row["measured_ms"]
                row["abs_delta_ms"] = d
                row["gap_pct"] = d / row["measured_ms"] * 100
            results.append(row)
            print()

    print(f"Completed in {time.time() - t0:.1f}s\n")

    # Save the consolidated results as JSON.
    summary_json = output_dir / "validation_summary.json"
    summary_json.write_text(json.dumps(results, indent=2))
    print(f"Per-row results saved to: {summary_json}")

    # Print a summary table grouped by (model, backend).
    if not any("predicted_ms" in r and "measured_ms" in r for r in results):
        print("(No rows have both predicted and measured values — nothing to compare.)")
        return 0

    print("\n" + "=" * 100)
    print("Validation summary (predicted vs measured)")
    print("=" * 100)
    # Group by (workload, model, backend) so prefill and decode get their
    # own headers + axis labels.
    by_combo: dict[tuple[str, str, str], list[dict]] = {}
    for r in results:
        wl = r.get("workload", "prefill")
        by_combo.setdefault((wl, r["model"], r["backend"]), []).append(r)

    for (wl, model, backend), rows in by_combo.items():
        rows.sort(key=lambda r: r["length"])
        axis_name = "L" if wl == "prefill" else "KV"
        header = (f"{axis_name:>6s}  {'measured ms':>13s}  {'± stddev':>10s}  "
                  f"{'predicted ms':>14s}  {'abs Δ ms':>10s}  {'gap %':>8s}")
        print(f"\n{model}  /  {backend}  /  {wl}")
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        for r in rows:
            if "predicted_ms" in r and "measured_ms" in r:
                std = r.get("measured_std_ms", 0.0)
                print(f"{r['length']:>6d}  "
                      f"{r['measured_ms']:>10.2f}    "
                      f"{std:>8.2f}  "
                      f"{r['predicted_ms']:>11.2f}    "
                      f"{r['abs_delta_ms']:>+8.2f}    "
                      f"{r['gap_pct']:>+6.1f}%")
            else:
                missing = []
                if "predicted_ms" not in r: missing.append("pred")
                if "measured_ms" not in r: missing.append("vllm")
                print(f"{r['length']:>6d}  missing: {', '.join(missing)}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
