#!/usr/bin/env python3
"""
Direct prediction test with vLLM comparison.

This script:
1. Measures actual vLLM execution time for batch configurations
2. Predicts latency using DoolySim CSV cache
3. Compares the results

No scheduler needed - just direct measurements and predictions!
"""

import sys
import time
import sqlite3
from pathlib import Path
from typing import List, Tuple, Dict

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Add simulator to path
simulator_dir = Path(__file__).parent.parent
if str(simulator_dir) not in sys.path:
    sys.path.insert(0, str(simulator_dir))

from simulator.core.prediction_cache import load_all_predictions
from simulator.operation_time_predictor import OperationTimePredictor

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM not available, will only show predictions")


def load_operation_counts(db_path: str, model_name: str, backend: str) -> Dict[Tuple[str, str], int]:
    """Load operation counts from the database."""
    conn = sqlite3.connect(db_path)
    query = """
        SELECT s.operation_name, s.signature_hash, mo.count
        FROM model_operations mo
        JOIN signatures s ON mo.operation_id = s.operation_id
        JOIN profiled_models pm ON mo.model_id = pm.model_id
        WHERE pm.model_name = ? AND pm.backend = ?
    """

    cursor = conn.execute(query, (model_name, backend))
    operations = {}
    for row in cursor:
        op_name, sig_hash, count = row
        operations[(op_name, sig_hash)] = count

    conn.close()
    return operations


def estimate_batch_latency(
    predictor: OperationTimePredictor,
    operation_counts: Dict[Tuple[str, str], int],
    batch_config: dict,
    verbose: bool = True
) -> float:
    """
    Estimate latency for a batch configuration.

    Args:
        predictor: OperationTimePredictor with CSV cache loaded
        operation_counts: Dictionary of (operation_name, sig_hash) -> count
        batch_config: Dictionary with batch parameters
        verbose: Whether to print details

    Returns:
        Estimated total latency in milliseconds
    """

    if verbose:
        print(f"\nBatch Config: {batch_config}")

    total_latency_ms = 0.0
    operation_contributions = []

    # Iterate through all operations
    for (op_name, sig_hash), count in operation_counts.items():
        sig_short = sig_hash[:8]
        model_key = f"{op_name}_{sig_short}"

        try:
            if op_name == "Attention":
                # Attention operations need special handling for prefill vs decode
                prefill_key = f"{op_name}_prefill_{sig_short}"
                decode_key = f"{op_name}_decode_{sig_short}"

                op_latency = 0.0

                # Prefill phase
                if batch_config.get('has_prefill', False):
                    agg_kv = batch_config.get('agg_kv_cache_prefill', 0)
                    agg_chunk = batch_config.get('agg_prefill_chunk', 0)

                    if prefill_key in predictor.predictions:
                        prefill_latency = predictor.predict_simple(
                            prefill_key,
                            kv_cache_size=agg_kv,
                            prefill_chunk_size=int(agg_chunk)
                        )
                        op_latency += prefill_latency

                        if verbose:
                            print(f"  {prefill_key}: {prefill_latency:.6f} ms")

                # Decode phase
                if batch_config.get('has_decode', False):
                    decode_batch = batch_config.get('decode_batch_size', 0)
                    avg_kv = batch_config.get('avg_kv_cache_decode', 0)

                    if decode_key in predictor.predictions:
                        decode_latency = predictor.predict_simple(
                            decode_key,
                            batch_size=decode_batch,
                            kv_cache_size=int(avg_kv)
                        )
                        op_latency += decode_latency

                        if verbose:
                            print(f"  {decode_key}: {decode_latency:.6f} ms")

                contribution = op_latency * count
                total_latency_ms += contribution

                if verbose and op_latency > 0:
                    print(f"    → {op_name}_{sig_short} subtotal: {op_latency:.6f} ms × {count} = {contribution:.6f} ms")

            else:
                # Non-attention operations
                if model_key in predictor.predictions:
                    total_tokens = batch_config.get('total_tokens', 0)

                    pred = predictor.predict_simple(
                        model_key,
                        num_tokens=total_tokens
                    )

                    contribution = pred * count
                    total_latency_ms += contribution

                    if verbose:
                        print(f"  {model_key}: {pred:.6f} ms × {count} = {contribution:.6f} ms")

                    operation_contributions.append((model_key, pred, count, contribution))

        except Exception as e:
            if verbose:
                print(f"  {model_key}: ERROR - {e}")

    return total_latency_ms


def create_test_cases() -> List[Tuple[str, dict]]:
    """Create test batch configurations."""

    return [
        # Test 1: Pure prefill (single request)
        (
            "Pure Prefill (single)",
            {
                'total_tokens': 32,
                'has_prefill': True,
                'has_decode': False,
                'agg_kv_cache_prefill': 0,
                'agg_prefill_chunk': 32,  # L2 norm = 128 for single request
            }
        ),
        # Test 2: Pure prefill (multiple requests)
        (
            "Pure Prefill (multiple)",
            {
                'total_tokens': 32,  #  1024 + 1024 + 1024
                'has_prefill': True,
                'has_decode': False,
                'agg_kv_cache_prefill': 0,
                'agg_prefill_chunk': (8**2 + 8**2 + 8**2 + 8**2)**0.5,  # L2 norm
            }
        ),
        # (
        #     "Pure Prefill (single)",
        #     {
        #         'total_tokens': 3072,  #  1024 + 1024 + 1024
        #         'has_prefill': True,
        #         'has_decode': False,
        #         'agg_kv_cache_prefill': 0,
        #         'agg_prefill_chunk': 3072,  # L2 norm
        #     }
        # ),
        # (
        #     "Decode (single)",
        #     {
        #         'total_tokens': 1,
        #         'has_prefill': False,
        #         'has_decode': True,
        #         'agg_kv_cache_prefill': 512,
        #         'agg_prefill_chunk': 0,  # L2 norm = 128 for single request
        #     }
        # ),

        # # Test 2: Pure prefill (multiple requests)
        # (
        #     "Pure Prefill (multiple)",
        #     {
        #         'total_tokens': 1,  #  1024 + 1024 + 1024
        #         'has_prefill': True,
        #         'has_decode': False,
        #         'agg_kv_cache_prefill': 0,
        #         'agg_prefill_chunk': (1024**2 + 1024**2 + 1024**2)**0.5,  # L2 norm
        #     }
        # ),
        # (
        #     "Pure Prefill (single)",
        #     {
        #         'total_tokens': 3072,  #  1024 + 1024 + 1024
        #         'has_prefill': True,
        #         'has_decode': False,
        #         'agg_kv_cache_prefill': 0,
        #         'agg_prefill_chunk': 3072,  # L2 norm
        #     }
        # ),
    ]


def measure_vllm_latency(llm, batch_config: dict, warmup: int = 3, iterations: int = 10) -> dict:
    """
    Measure actual vLLM latency for a batch configuration.

    Args:
        llm: vLLM LLM instance
        batch_config: Batch configuration dict
        warmup: Number of warmup iterations
        iterations: Number of measurement iterations

    Returns:
        Dictionary with timing statistics
    """
    # Create prompts based on batch config
    prompts = []
    max_tokens_per_prompt = []

    # For prefill requests
    if batch_config.get('has_prefill', False):
        # Extract prefill chunk size - for simplicity, use sqrt of L2 norm
        agg_chunk = batch_config.get('agg_prefill_chunk', 128)
        # Approximate individual sizes - for simplicity, use uniform
        num_prefill = 1  # We'll create one prefill prompt with aggregate size
        prompt_len = int(agg_chunk)
        prompts.append(" ".join(["token"] * prompt_len))
        max_tokens_per_prompt.append(1 if batch_config.get('has_decode', False) else 0)

    # For decode requests
    if batch_config.get('has_decode', False):
        decode_batch = batch_config.get('decode_batch_size', 1)
        avg_kv = batch_config.get('avg_kv_cache_decode', 512)

        # Create decode prompts (already prefilled, just generating 1 token)
        for _ in range(decode_batch):
            prompts.append(" ".join(["token"] * avg_kv))
            max_tokens_per_prompt.append(1)

    # Use max across all prompts for simplicity
    max_tokens = max(max_tokens_per_prompt) if max_tokens_per_prompt else 1

    # Sampling params
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        ignore_eos=True,
    )

    # Warmup
    for _ in range(warmup):
        llm.generate(prompts, sampling_params, use_tqdm=False)

    # Measure
    latencies = []
    if TORCH_AVAILABLE:
        torch.cuda.synchronize()

    for _ in range(iterations):
        start = time.perf_counter()
        llm.generate(prompts, sampling_params, use_tqdm=False)
        if TORCH_AVAILABLE:
            torch.cuda.synchronize()
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)

    return {
        'mean_ms': sum(latencies) / len(latencies),
        'min_ms': min(latencies),
        'max_ms': max(latencies),
        'median_ms': sorted(latencies)[len(latencies) // 2],
        'std_ms': (sum((x - sum(latencies)/len(latencies))**2 for x in latencies) / len(latencies))**0.5,
        'all_latencies': latencies,
    }


def main():
    """Main test entry point."""

    import argparse
    parser = argparse.ArgumentParser(description="Test DoolySim predictions vs vLLM")
    parser.add_argument('--with-vllm', action='store_true', help='Measure actual vLLM latencies')
    parser.add_argument('--warmup', type=int, default=3, help='Number of warmup iterations')
    parser.add_argument('--iterations', type=int, default=10, help='Number of measurement iterations')
    args = parser.parse_args()

    # Configuration
    MODEL = "meta-llama/Llama-3.1-8B"
    CACHE_DIR = "/home/jkim38/dooly/cache/tp1/meta-llama/Llama-3.1-8B/csv_predictions/"
    DB_PATH = "/home/jkim38/dooly/profiler/database/dooly_data_new_new.db"
    BACKEND = "FLASHINFER"

    print("=" * 80)
    print("DIRECT PREDICTION TEST (NO SCHEDULER)")
    print("=" * 80)
    print(f"Model:    {MODEL}")
    print(f"Backend:  {BACKEND}")
    print(f"Cache:    {CACHE_DIR}")
    print("=" * 80)
    print()

    # Step 1: Load CSV cache
    print("Step 1: Loading CSV predictions...")
    cache = load_all_predictions(CACHE_DIR)
    print(f"        Loaded {len(cache.predictions)} operations")
    print()

    # Step 2: Load operation counts
    print("Step 2: Loading operation counts from database...")
    operation_counts = load_operation_counts(DB_PATH, MODEL, BACKEND)
    print(f"        Loaded {len(operation_counts)} operation counts")
    print()

    # Show operation counts
    print("Operation counts:")
    for (op_name, sig_hash), count in sorted(operation_counts.items(), key=lambda x: -x[1]):
        print(f"  {op_name:30s} {sig_hash[:8]} × {count:3d}")
    print()

    # Step 3: Create predictor
    print("Step 3: Creating predictor...")
    operation_is_context = {}
    for op_name in cache.predictions.keys():
        base_name = op_name.replace('_prefill', '').replace('_decode', '')
        import re
        base_name = re.sub(r'_[0-9a-f]{8}$', '', base_name)
        operation_is_context[base_name] = 'Attention' in base_name

    predictor = OperationTimePredictor(
        models={},
        predictions=cache.predictions,
        operation_is_context=operation_is_context,
        kv_cache_granularity=64,
    )
    print("        Predictor ready")
    print()

    # Step 4: Optionally initialize vLLM
    llm = None
    if args.with_vllm:
        if not VLLM_AVAILABLE:
            print("ERROR: vLLM not available but --with-vllm was specified")
            return

        print("Step 4: Initializing vLLM engine...")
        llm = LLM(
            model=MODEL,
            gpu_memory_utilization=0.9,
            max_num_seqs=256,
            max_model_len=4096,
            enforce_eager=True,  # Disable CUDA graphs for accurate timing
        )
        print("        vLLM engine ready")
        print()

    # Step 5: Run test cases
    print("=" * 80)
    print("RUNNING TEST CASES")
    print("=" * 80)

    test_cases = create_test_cases()
    results = []

    for test_name, batch_config in test_cases:
        print()
        print(f"TEST: {test_name}")
        print("-" * 80)

        # DoolySim prediction
        dooly_latency = estimate_batch_latency(
            predictor=predictor,
            operation_counts=operation_counts,
            batch_config=batch_config,
            verbose=False  # Set to True for detailed output
        )

        print(f"  DoolySim prediction: {dooly_latency:.3f} ms")

        # vLLM measurement
        vllm_stats = None
        if llm is not None:
            print(f"  Measuring vLLM (warmup={args.warmup}, iterations={args.iterations})...")
            try:
                vllm_stats = measure_vllm_latency(
                    llm=llm,
                    batch_config=batch_config,
                    warmup=args.warmup,
                    iterations=args.iterations
                )
                print(f"  vLLM actual:         {vllm_stats['mean_ms']:.3f} ± {vllm_stats['std_ms']:.3f} ms")
                print(f"                       (min: {vllm_stats['min_ms']:.3f}, max: {vllm_stats['max_ms']:.3f})")

                # Calculate error
                error_pct = abs(dooly_latency - vllm_stats['mean_ms']) / vllm_stats['mean_ms'] * 100
                print(f"  Error:               {error_pct:.1f}%")

            except Exception as e:
                print(f"  vLLM measurement failed: {e}")

        print("-" * 80)

        results.append({
            'test_name': test_name,
            'tokens': batch_config['total_tokens'],
            'dooly_ms': dooly_latency,
            'vllm_ms': vllm_stats['mean_ms'] if vllm_stats else None,
            'error_pct': abs(dooly_latency - vllm_stats['mean_ms']) / vllm_stats['mean_ms'] * 100 if vllm_stats else None,
        })

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if llm is not None:
        print(f"{'Test Name':<35} {'Tokens':>8} {'DoolySim':>12} {'vLLM':>12} {'Error':>10}")
        print("-" * 80)
        for r in results:
            vllm_str = f"{r['vllm_ms']:.3f}" if r['vllm_ms'] is not None else "N/A"
            error_str = f"{r['error_pct']:.1f}%" if r['error_pct'] is not None else "N/A"
            print(f"{r['test_name']:<35} {r['tokens']:>8} {r['dooly_ms']:>12.3f} {vllm_str:>12} {error_str:>10}")

        # Calculate average error
        errors = [r['error_pct'] for r in results if r['error_pct'] is not None]
        if errors:
            print("-" * 80)
            print(f"{'Average Error:':<35} {sum(errors)/len(errors):>10.1f}%")
    else:
        print(f"{'Test Name':<35} {'Tokens':>8} {'DoolySim (ms)':>15}")
        print("-" * 80)
        for r in results:
            print(f"{r['test_name']:<35} {r['tokens']:>8} {r['dooly_ms']:>15.3f}")

    print("=" * 80)


if __name__ == "__main__":
    main()
