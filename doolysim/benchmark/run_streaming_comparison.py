#!/usr/bin/env python3
"""Run DoolySim (doolysim.cli) and vLLM streaming (doolysim.benchmark.measure_vllm_streaming)
across multiple configurations on the same dataset, writing matched JSON outputs
into a `comparison/<run>/<model>/` layout that `plot_configs_cdf.py` can consume.

Edit the CONFIGS list (or pass --config) and the common settings at the top of
main(), then:

    python -m doolysim.benchmark.run_streaming_comparison \\
        --dataset ./dataset/openchat-1.0rps.jsonl \\
        --run-name openchat-1.0rps \\
        --results-root ./comparison

Skip one side with --skip-dooly or --skip-vllm.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class RunConfig:
    model: str
    tp: int
    backend: str
    cache_dir: str
    topology: str
    db_path: str
    quantization: Optional[str] = None  # e.g. "fp8" — passed only to the vLLM streaming side


DEFAULT_CONFIGS: List[RunConfig] = [
    RunConfig(
        model="meta-llama/Llama-3.1-8B",
        tp=1,
        backend="FLASHINFER",
        cache_dir="/home/ubuntu/dooly/h100-workspace/cache/meta-llama/Llama-3.1-8B/",
        topology="mew1",
        db_path="/home/ubuntu/dooly/h100-workspace/database/h100-sharegpt-eval.db",
    ),
    
    # RunConfig(
    #     model="Qwen/Qwen3-30B-A3B",
    #     tp=1,
    #     backend="FLASHINFER",
    #     cache_dir="/home/ubuntu/dooly/h100-workspace/cache/Qwen/Qwen3-30B-A3B",
    #     topology="mew1",
    #     db_path="/home/ubuntu/dooly/h100-workspace/database/h100-sharegpt-eval.db",
    # ),
    RunConfig(
        model="Qwen/Qwen3-32B",
        tp=1,
        backend="FLASHINFER",
        cache_dir="/home/ubuntu/dooly/h100-workspace/cache/Qwen/Qwen3-32B/",
        topology="mew1",
        db_path="/home/ubuntu/dooly/h100-workspace/database/h100-sharegpt-eval.db",
    ),
    # RunConfig(
    #     model="CohereLabs/c4ai-command-r7b-12-2024",
    #     tp=1,
    #     backend="FLASHINFER",
    #     cache_dir="/home/ubuntu/dooly/h100-workspace/cache/CohereLabs/c4ai-command-r7b-12-2024",
    #     topology="mew1",
    #     db_path="/home/ubuntu/dooly/h100-workspace/database/h100-sharegpt-eval.db",
    # ),
    RunConfig(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        tp=1,
        backend="FLASHINFER",
        cache_dir="/home/ubuntu/dooly/h100-workspace/cache/mistralai/Mixtral-8x7B-Instruct-v0.1",
        topology="mew1",
        db_path="/home/ubuntu/dooly/h100-workspace/database/test-quant.db",
        quantization="fp8",
    ),
    RunConfig(
        model="mistralai/Ministral-8B-Instruct-2410",
        tp=1,
        backend="FLASHINFER",
        cache_dir="/home/ubuntu/dooly/h100-workspace/cache/mistralai/Ministral-8B-Instruct-2410",
        topology="mew1",
        db_path="/home/ubuntu/dooly/h100-workspace/database/h100-sharegpt-eval.db",
    ), 
    RunConfig(
        model="Qwen/Qwen3-14B",
        tp=1,
        backend="FLASHINFER",
        cache_dir="/home/ubuntu/dooly/h100-workspace/cache/Qwen/Qwen3-14B/",
        topology="mew1",
        db_path="/home/ubuntu/dooly/h100-workspace/database/h100-sharegpt-eval.db",
    ),
]

def run_command(cmd: List[str], description: str, log_file: Optional[str] = None) -> bool:
    banner = "=" * 80
    print(f"\n{banner}\n{description}\nCommand: {' '.join(cmd)}\n{banner}")
    try:
        if log_file:
            with open(log_file, "w") as lf:
                subprocess.run(cmd, check=True, stdout=lf, stderr=subprocess.STDOUT)
            print(f"✓ Success (log: {log_file})")
        else:
            subprocess.run(cmd, check=True)
            print("✓ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed with exit code {e.returncode}")
        return False


def dooly_cmd(
    cfg: RunConfig,
    dataset: str,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    max_model_len: int,
    gpu_mem_util: float,
    seed: int,
    json_output: str,
    batch_composition_output: str,
    verbose: bool,
) -> List[str]:
    cmd = [
        sys.executable, "-m", "doolysim.cli",
        "--model", cfg.model,
        "--backend", cfg.backend,
        "--tp", str(cfg.tp),
        "--topology", cfg.topology,
        "--cache-dir", cfg.cache_dir,
        "--db-path", cfg.db_path,
        "--load-dataset", dataset,
        "--max-num-batched-tokens", str(max_num_batched_tokens),
        "--max-num-seqs", str(max_num_seqs),
        "--max-model-len", str(max_model_len),
        "--gpu-mem-util", str(gpu_mem_util),
        "--seed", str(seed),
        "--export-json",
        "--json-output", json_output,
        "--export-batch-composition",
        "--batch-composition-output", batch_composition_output,
    ]
    if verbose:
        cmd.append("--verbose")
    return cmd


def vllm_cmd(
    cfg: RunConfig,
    dataset: str,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    max_model_len: int,
    gpu_mem_util: float,
    seed: int,
    output: str,
    batch_composition_output: str,
    verbose: bool,
) -> List[str]:
    cmd = [
        sys.executable, "-m", "doolysim.benchmark.measure_vllm_streaming",
        "--model", cfg.model,
        "--backend", cfg.backend,
        "--tp", str(cfg.tp),
        "--load-dataset", dataset,
        "--max-num-batched-tokens", str(max_num_batched_tokens),
        "--max-num-seqs", str(max_num_seqs),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--seed", str(seed),
        "--output", output,
        "--batch-composition-output", batch_composition_output,
    ]
    if cfg.quantization:
        cmd.extend(["--quantization", cfg.quantization])
    if verbose:
        cmd.append("--verbose")
    return cmd


def run_for_config(
    cfg: RunConfig,
    dataset: str,
    run_name: str,
    results_root: str,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    max_model_len: int,
    gpu_mem_util: float,
    seed: int,
    skip_dooly: bool,
    skip_vllm: bool,
    verbose: bool,
) -> dict:
    out_dir = Path(results_root) / run_name / cfg.model
    out_dir.mkdir(parents=True, exist_ok=True)

    dooly_json = str(out_dir / f"dooly_{run_name}.json")
    dooly_batch = str(out_dir / f"dooly_batch_{run_name}.json")
    vllm_json = str(out_dir / f"vllm_{run_name}.json")
    vllm_batch = str(out_dir / f"vllm_batch_{run_name}.json")

    dooly_ok = True
    if not skip_dooly:
        dooly_ok = run_command(
            dooly_cmd(
                cfg, dataset, max_num_batched_tokens, max_num_seqs, max_model_len,
                gpu_mem_util, seed, dooly_json, dooly_batch, verbose,
            ),
            f"DoolySim | {cfg.model} (tp={cfg.tp}, {cfg.backend})",
            log_file=str(out_dir / f"dooly_{run_name}.log"),
        )

    vllm_ok = True
    if not skip_vllm:
        vllm_ok = run_command(
            vllm_cmd(
                cfg, dataset, max_num_batched_tokens, max_num_seqs, max_model_len,
                gpu_mem_util, seed, vllm_json, vllm_batch, verbose,
            ),
            f"vLLM streaming | {cfg.model} (tp={cfg.tp}, {cfg.backend})",
            log_file=str(out_dir / f"vllm_{run_name}.log"),
        )

    return {
        "model": cfg.model,
        "tp": cfg.tp,
        "backend": cfg.backend,
        "out_dir": str(out_dir),
        "dooly_ok": dooly_ok and not skip_dooly,
        "vllm_ok": vllm_ok and not skip_vllm,
        "dooly_skipped": skip_dooly,
        "vllm_skipped": skip_vllm,
    }


def filter_configs(configs: List[RunConfig], only_models: Optional[List[str]]) -> List[RunConfig]:
    if not only_models:
        return configs
    wanted = set(only_models)
    kept = [c for c in configs if c.model in wanted]
    missing = wanted - {c.model for c in kept}
    if missing:
        print(f"WARNING: unknown models in --only: {sorted(missing)}")
    return kept


def main() -> None:
    # Make sure ninja / nvcc are reachable for FlashInfer JIT inside vLLM.
    # Same fix as run_comparison.py — when invoked via the absolute conda
    # python, the env's bin/ isn't on PATH and EngineCore can't find ninja.
    py_bindir = os.path.dirname(sys.executable)
    if py_bindir not in os.environ.get("PATH", "").split(":"):
        os.environ["PATH"] = f"{py_bindir}:{os.environ.get('PATH', '')}"
    for cuda_bin in ("/usr/local/cuda-13.2/bin", "/usr/local/cuda-12.8/bin"):
        if os.path.isdir(cuda_bin) and cuda_bin not in os.environ["PATH"]:
            os.environ["PATH"] = f"{cuda_bin}:{os.environ['PATH']}"
            os.environ.setdefault("CUDA_HOME", os.path.dirname(cuda_bin))
            break

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True,
                        help="Path to the JSONL dataset (e.g. ./dataset/openchat-0.5rps.jsonl)")
    parser.add_argument("--run-name", required=True,
                        help="Subdir name under --results-root (e.g. openchat-0.5rps)")
    parser.add_argument("--results-root", default="./comparison",
                        help="Root directory for per-run outputs (default: ./comparison)")
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-mem-util", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", nargs="+",
                        help="Restrict to these model names (must match CONFIGS exactly)")
    parser.add_argument("--skip-dooly", action="store_true")
    parser.add_argument("--skip-vllm", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not os.path.isfile(args.dataset):
        parser.error(f"dataset not found: {args.dataset}")
    if args.skip_dooly and args.skip_vllm:
        parser.error("--skip-dooly and --skip-vllm can't both be set")

    print(args.only)
    configs = filter_configs(DEFAULT_CONFIGS, args.only)
    if not configs:
        parser.error("no configs selected (check --only)")

    print(f"Running {len(configs)} config(s) on dataset {args.dataset}")
    print(f"Results -> {Path(args.results_root) / args.run_name}")

    results = []
    for cfg in configs:
        results.append(run_for_config(
            cfg=cfg,
            dataset=args.dataset,
            run_name=args.run_name,
            results_root=args.results_root,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            max_model_len=args.max_model_len,
            gpu_mem_util=args.gpu_mem_util,
            seed=args.seed,
            skip_dooly=args.skip_dooly,
            skip_vllm=args.skip_vllm,
            verbose=args.verbose,
        ))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for r in results:
        dooly = "skip" if r["dooly_skipped"] else ("✓" if r["dooly_ok"] else "✗")
        vllm = "skip" if r["vllm_skipped"] else ("✓" if r["vllm_ok"] else "✗")
        print(f"  {r['model']:<40s} tp={r['tp']}  dooly={dooly}  vllm={vllm}  -> {r['out_dir']}")


if __name__ == "__main__":
    main()
