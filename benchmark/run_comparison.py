#!/usr/bin/env python3
"""
Script to run batch predictor and vLLM direct measurements across multiple configurations.
"""

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict

def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Success")
        if result.stdout:
            print("STDOUT:", result.stdout[-500:])  # Show last 500 chars
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed with exit code {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout[-500:])
        if e.stderr:
            print("STDERR:", e.stderr[-500:])
        return False

def generate_batch_configs(output_dir: str) -> List[str]:
    """Generate different batch configuration files."""
    configs = []

    # Different batch scenarios
    scenarios = [
        {
            "name": "small_prefill_256",
            "description": "Small prefill batch",
            "prefill_requests": [
                {"tokens": 256, "kv_cache": 0},
            ],
            "decode_requests": []
        },
        {
            "name": "small_prefill_512",
            "description": "Small prefill batch",
            "prefill_requests": [
                {"tokens": 512, "kv_cache": 0},
            ],
            "decode_requests": []
        },
        {
            "name": "small_prefill_1024",
            "description": "Small prefill batch",
            "prefill_requests": [
                {"tokens": 1024, "kv_cache": 0},
            ],
            "decode_requests": []
        },
        {
            "name": "small_prefill_2048",
            "description": "Small prefill batch",
            "prefill_requests": [
                {"tokens": 2048, "kv_cache": 0},
            ],
            "decode_requests": []
        },
        {
            "name": "small_prefill_4096",
            "description": "Small prefill batch",
            "prefill_requests": [
                {"tokens": 4096, "kv_cache": 0},
            ],
            "decode_requests": []
        },
    ]

    os.makedirs(output_dir, exist_ok=True)

    for scenario in scenarios:
        config_path = os.path.join(output_dir, f"batch_{scenario['name']}.json")
        config_data = {
            "description": scenario["description"],
            "model": "meta-llama/Llama-3.1-8B",
            "backend": "FLASHINFER",
            "tensor_parallel_size": 2,
            "prefill_requests": scenario["prefill_requests"],
            "decode_requests": scenario["decode_requests"]
        }

        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)

        configs.append(config_path)
        print(f"Generated config: {config_path}")

    return configs

def main():
    # Configuration
    CACHE_DIR = "./cache/tp2/meta-llama/Llama-3.1-8B/"
    DB_PATH = "/home/jkim38/dooly/profiler/database/dooly.db"
    RESULTS_DIR = "./results/"
    CONFIGS_DIR = "./batch_configs/"
    WARMUP = 3
    ITERATIONS = 4

    print("Batch Comparison Runner")
    print("=" * 80)

    # Create output directories
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Clean up vLLM profiles directory to prevent reading stale traces
    VLLM_PROFILES_DIR = "./vllm_profiles/"
    if os.path.exists(VLLM_PROFILES_DIR):
        shutil.rmtree(VLLM_PROFILES_DIR)
        print(f"Cleaned up {VLLM_PROFILES_DIR} directory to prevent stale trace reads")

    # Generate batch configurations
    print("\n1. Generating batch configurations...")
    config_files = generate_batch_configs(CONFIGS_DIR)

    # Run comparisons for each configuration
    print(f"\n2. Running comparisons for {len(config_files)} configurations...")

    results_summary = []

    for config_file in config_files:
        config_name = Path(config_file).stem
        print(f"\n{'='*80}")
        print(f"Processing: {config_name}")
        print(f"Config: {config_file}")
        print(f"{'='*80}")

        # Output file names
        dooly_output = os.path.join(RESULTS_DIR, f"dooly_{config_name}.json")
        vllm_output = os.path.join(RESULTS_DIR, f"vllm_{config_name}.json")

        # Run Dooly batch predictor
        dooly_cmd = [
            sys.executable, "-m", "simulator.batch_predictor",
            "--batch-config", config_file,
            "--cache-dir", CACHE_DIR,
            "--db-path", DB_PATH,
            "--json-output", dooly_output
        ]

        dooly_success = run_command(dooly_cmd, f"Dooly prediction for {config_name}")

        # Run vLLM direct measurement
        vllm_cmd = [
            sys.executable, "-m", "simulator.benchmark.measure_vllm_direct",
            "--batch-config", config_file,
            "--warmup", str(WARMUP),
            "--iterations", str(ITERATIONS),
            "--max-model-len", "8194",
            "--max-num-seqs", "256",
            "--gpu-memory", "0.9",
            "--out-dir", RESULTS_DIR,
            "--output", f"vllm_{config_name}.json"
        ]

        vllm_success = run_command(vllm_cmd, f"vLLM measurement for {config_name}")

        # Track results
        result_entry = {
            "config": config_name,
            "config_file": config_file,
            "dooly_output": dooly_output if dooly_success else None,
            "vllm_output": vllm_output if vllm_success else None,
            "dooly_success": dooly_success,
            "vllm_success": vllm_success
        }
        results_summary.append(result_entry)

    # Generate summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    successful_pairs = 0
    for result in results_summary:
        status = "✓" if result["dooly_success"] and result["vllm_success"] else "✗"
        print(f"{status} {result['config']}: Dooly={'✓' if result['dooly_success'] else '✗'}, vLLM={'✓' if result['vllm_success'] else '✗'}")
        if result["dooly_success"] and result["vllm_success"]:
            successful_pairs += 1

    print(f"\nSuccessful pairs: {successful_pairs}/{len(results_summary)}")

    # Save summary
    summary_file = os.path.join(RESULTS_DIR, "comparison_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"Summary saved to: {summary_file}")

    # Suggest next steps
    if successful_pairs > 0:
        print(f"\nNext steps:")
        print(f"1. Compare results: python -m simulator.benchmark.compare_json <dooly_file> <vllm_file>")
        print(f"2. Analyze all results in: {RESULTS_DIR}")

if __name__ == "__main__":
    main()