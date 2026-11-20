#!/usr/bin/env python3
"""
Long-context sweep comparison runner.

For each (model, backend, scenario) tuple, generates a batch config and runs
both Dooly's batch predictor and vLLM's direct measurement, producing a
side-by-side dataset suitable for the §7 long-context evaluation figures
(TTFT vs input length, per-step decode latency vs KV-cache size).

Sweep dimensions (edit MODELS / BACKENDS / lengths at the top):
  models   : meta-llama/Llama-3.1-8B, CohereLabs/c4ai-command-r7b-12-2024
  backends : FLASH_ATTN, TRITON_ATTN  (add FLASHINFER if desired)
  prefill  : 512, 1024, 2048, 4096, 8192, 16384  (input length)
  decode   : 1024, 2048, 4096, 8192, 16384       (KV-cache size for single decode step)

The 4K boundary is intentionally crossed so Command-R7B's interleaved
sliding-window attention (window=4096) plateau emerges in the curves.
"""

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict


# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

MODELS = [
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
]

BACKENDS = ["FLASHINFER"]

# Quantization scheme to apply to all runs (None = BF16). When set to "fp8",
# both batch_predictor (uses the FP8-profiled DB) and measure_vllm_direct
# (loads the model with quantization=fp8) operate consistently.
QUANTIZATION = "fp8"

PREFILL_LENGTHS = [512, 1024, 2048]
DECODE_KV_SIZES = []

# Per-run knobs
WARMUP = 0
ITERATIONS = 1        # take median across iterations for noise reduction
MAX_MODEL_LEN = 4096  # must exceed the largest scenario length; keep small for FP8 KV-cache budget
MAX_NUM_SEQS = 32
GPU_MEMORY = 0.9

DB_PATH = "/home/ubuntu/dooly/test-quant.db"
CACHE_ROOT = "/home/ubuntu/dooly/h100-workspace/cache"   # cache layout: {CACHE_ROOT}/{model}/{backend}
RESULTS_DIR = "./results/mixtral_fp8_sweep"
CONFIGS_DIR = "./batch_configs/mixtral_fp8_sweep"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_name(model: str) -> str:
    """Replace path separators so model names are filesystem-safe."""
    return model.replace("/", "__")


def cache_dir_for(model: str, backend: str) -> str:
    """Cache layout: {CACHE_ROOT}/{model}/  (per-backend subdir is unused)."""
    return os.path.join(CACHE_ROOT, model)


def build_scenarios() -> List[Dict]:
    """Construct the scenario list shared across all (model, backend) pairs."""
    scenarios: List[Dict] = []
    for length in PREFILL_LENGTHS:
        scenarios.append({
            "name": f"prefill_{length}",
            "description": f"Single prefill batch, input_len={length}",
            "prefill_requests": [{"tokens": length, "kv_cache": 0}],
            "decode_requests": [],
        })
    for kv in DECODE_KV_SIZES:
        scenarios.append({
            "name": f"decode_kv_{kv}",
            "description": f"Single decode step at KV={kv}",
            "prefill_requests": [],
            "decode_requests": [{"tokens": 1, "kv_cache": kv}],
        })
    return scenarios


def write_config(model: str, backend: str, scenario: Dict, configs_dir: str) -> str:
    """Write one batch-config JSON for a (model, backend, scenario) tuple."""
    config_path = os.path.join(
        configs_dir,
        f"{safe_name(model)}__{backend}__{scenario['name']}.json",
    )
    config_data = {
        "description": f"{model} | {backend} | {scenario['description']}",
        "model": model,
        "backend": backend,
        "tensor_parallel_size": 1,
        "prefill_requests": scenario["prefill_requests"],
        "decode_requests": scenario["decode_requests"],
    }
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
    return config_path


def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and return success status (stdout/stderr tail on failure)."""
    print(f"\n{'=' * 80}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 80}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Success")
        if result.stdout:
            print("STDOUT:", result.stdout[-500:])
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed with exit code {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout[-500:])
        if e.stderr:
            print("STDERR:", e.stderr[-500:])
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Make sure ninja / nvcc are reachable for FlashInfer JIT inside vLLM.
    # When this script is invoked with the absolute conda python path, the
    # interpreter's bin dir (which contains `ninja`) isn't on PATH, and
    # vLLM's EngineCore subprocess fails when compiling FP8 kernels.
    py_bindir = os.path.dirname(sys.executable)
    if py_bindir not in os.environ.get("PATH", "").split(":"):
        os.environ["PATH"] = f"{py_bindir}:{os.environ.get('PATH', '')}"
    for cuda_bin in ("/usr/local/cuda-13.2/bin", "/usr/local/cuda-12.8/bin"):
        if os.path.isdir(cuda_bin) and cuda_bin not in os.environ["PATH"]:
            os.environ["PATH"] = f"{cuda_bin}:{os.environ['PATH']}"
            os.environ.setdefault("CUDA_HOME", os.path.dirname(cuda_bin))
            break

    print("Long-context sweep comparison runner")
    print("=" * 80)
    print(f"  models  : {MODELS}")
    print(f"  backends: {BACKENDS}")
    print(f"  prefill : {PREFILL_LENGTHS}")
    print(f"  decode  : {DECODE_KV_SIZES}")
    print(f"  iters   : warmup={WARMUP}, measurement={ITERATIONS}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CONFIGS_DIR, exist_ok=True)

    # Clean stale vLLM trace artefacts so subsequent reads aren't polluted.
    vllm_profiles_dir = "./vllm_profiles/"
    if os.path.exists(vllm_profiles_dir):
        shutil.rmtree(vllm_profiles_dir)
        print(f"Cleaned {vllm_profiles_dir} (stale traces).")

    scenarios = build_scenarios()
    print(f"\n1. Built {len(scenarios)} scenarios "
          f"({len(PREFILL_LENGTHS)} prefill + {len(DECODE_KV_SIZES)} decode).")

    total_runs = len(MODELS) * len(BACKENDS) * len(scenarios)
    print(f"2. Will execute {total_runs} (model × backend × scenario) runs.\n")

    results_summary = []
    run_idx = 0
    for model in MODELS:
        for backend in BACKENDS:
            cache_dir = cache_dir_for(model, backend)
            os.makedirs(cache_dir, exist_ok=True)

            for scenario in scenarios:
                run_idx += 1
                tag = f"{safe_name(model)}__{backend}__{scenario['name']}"
                print(f"\n[{run_idx}/{total_runs}] {tag}")

                config_file = write_config(model, backend, scenario, CONFIGS_DIR)
                dooly_output = os.path.join(RESULTS_DIR, f"dooly_{tag}.json")
                vllm_output_name = f"vllm_{tag}.json"
                vllm_output = os.path.join(RESULTS_DIR, vllm_output_name)

                # Dooly prediction
                dooly_cmd = [
                    sys.executable, "-m", "doolysim.benchmark.batch_predictor",
                    "--batch-config", config_file,
                    "--cache-dir", cache_dir,
                    "--db-path", DB_PATH,
                    "--json-output", dooly_output,
                ]
                dooly_success = run_command(dooly_cmd, f"Dooly: {tag}")

                # vLLM measurement
                vllm_cmd = [
                    sys.executable, "-m", "doolysim.benchmark.measure_vllm_direct",
                    "--batch-config", config_file,
                    "--warmup", str(WARMUP),
                    "--iterations", str(ITERATIONS),
                    "--max-model-len", str(MAX_MODEL_LEN),
                    "--max-num-seqs", str(MAX_NUM_SEQS),
                    "--gpu-memory", str(GPU_MEMORY),
                    "--out-dir", RESULTS_DIR,
                    "--output", vllm_output_name,
                ]
                if QUANTIZATION:
                    vllm_cmd.extend(["--quantization", QUANTIZATION])
                vllm_success = run_command(vllm_cmd, f"vLLM: {tag}")

                results_summary.append({
                    "tag": tag,
                    "model": model,
                    "backend": backend,
                    "scenario": scenario["name"],
                    "config_file": config_file,
                    "dooly_output": dooly_output if dooly_success else None,
                    "vllm_output": vllm_output if vllm_success else None,
                    "dooly_success": dooly_success,
                    "vllm_success": vllm_success,
                })

    # Final summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    successful_pairs = 0
    for r in results_summary:
        d = "✓" if r["dooly_success"] else "✗"
        v = "✓" if r["vllm_success"] else "✗"
        status = "✓" if r["dooly_success"] and r["vllm_success"] else "✗"
        print(f"  {status} {r['tag']:<80s}  Dooly={d}  vLLM={v}")
        if r["dooly_success"] and r["vllm_success"]:
            successful_pairs += 1

    print(f"\nSuccessful pairs: {successful_pairs}/{len(results_summary)}")

    summary_file = os.path.join(RESULTS_DIR, "comparison_summary.json")
    with open(summary_file, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"Summary saved to: {summary_file}")

    if successful_pairs > 0:
        print(f"\nNext steps:")
        print(f"  1. Aggregate per-(model, backend) into the line plots described in §7.")
        print(f"  2. All raw outputs are under: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

