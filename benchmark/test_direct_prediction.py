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
import os 

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

def load_model_collectives(db_path, backend, model_name, topology, tp_size):
    conn = sqlite3.connect(db_path)
    query = """
        SELECT co.operation_name, co.dim_param, co.num_dims, mc.count
        FROM model_collectives mc
        JOIN coll_ops co ON mc.collective_id = co.collective_id
        JOIN profiled_models pm ON mc.model_id = pm.model_id
        WHERE pm.model_name = ?
            AND pm.backend = ?
            AND co.topology = ?
            AND co.tp_degree = ?
    """

    cursor = conn.execute(query, (model_name, backend, topology, tp_size))
    collectives = cursor.fetchall()
    conn.close()

    if collectives:
        print(f"Loaded {len(collectives)} collective operations (topology={topology}, tp={tp_size})")
        for op_name, dim_param, num_dims, count in collectives:
            print(f"  - {op_name} (dim={dim_param}, ndims={num_dims}) × {count}")
    else:
        print(f"No collective operations found for {model_name} (topology={topology}, tp={tp_size})")

    print(collectives)
    return collectives

def estimate_comm_latency(
    predictor: OperationTimePredictor,
    configs,
    model_collectives,
    batch
) -> float:
    comm_latency_ms = 0.0
    dtype_bytes = 2
    
    def calculate_collective_size(op_name, dim_param, num_dims, batch):
        # Look up configuration
        key = (op_name, dim_param, num_dims)
        config = configs.get(key)

        # Use stored configuration
        batch_param = config['batch_param']
        model_config_dim = config['model_config_dim']

        if batch_param == "NUM_TOKS":
            num_elements = batch['total_tokens']
        elif batch_param == "NUM_REQS":
            num_elements = batch['num_requests']  # Number of requests in batch
        else:
            raise ValueError(f"Unknown batch_param: {batch_param}")

        size_bytes = num_elements * model_config_dim * dtype_bytes

        return size_bytes
    
    def estimate_collective_latency(op_name, dim_param, num_dims, size_bytes):
        dim_str = "None" if dim_param is None or (isinstance(dim_param, float) and np.isnan(dim_param)) else str(int(dim_param))
        
        if op_name == 'all_reduce':
            num_dims = 1
        model_key = f"{op_name}_dim{dim_str}_ndims{int(num_dims)}"

        try:
            # Check if model exists in either models or predictions (CSV cache)
            if model_key not in predictor.models and model_key not in predictor.predictions:
                print(f"  Warning: No model found for {model_key}")
                return 0.0

            # Use predict_simple with size_bytes feature
            pred = predictor.predict_simple(
                model_key,
                size_bytes=size_bytes
            )
            return float(pred)

        except Exception as e:
            print(f"  Warning: Model prediction failed for {model_key}: {e}")
            return 0.0
    
    for op_name, dim_param, num_dims, count in model_collectives:
        # Calculate tensor size for this batch using model-specific config
        size_bytes = calculate_collective_size(op_name, dim_param, num_dims, batch)

        # Estimate latency for one occurrence
        op_latency = estimate_collective_latency(op_name, dim_param, num_dims, size_bytes)

        # Multiply by count
        contribution = op_latency * count
        comm_latency_ms += contribution
    
    return comm_latency_ms

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
                        num_tokens=total_tokens,
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
                'total_tokens': 256,
                'has_prefill': True,
                'has_decode': False,
                'agg_kv_cache_prefill': 0,
                'agg_prefill_chunk': 256,  # L2 norm = 128 for single request
                'num_requests': 1
            }
        ),
        # Test 2: Pure prefill (multiple requests)
        # (
        #     "Pure Prefill (multiple)",
        #     {
        #         'total_tokens': 32,  #  1024 + 1024 + 1024
        #         'has_prefill': True,
        #         'has_decode': False,
        #         'agg_kv_cache_prefill': 0,
        #         'agg_prefill_chunk': (8**2 + 8**2 + 8**2 + 8**2)**0.5,  # L2 norm
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

def load_configs(db_path, model_name, backend, tp_size, topology) -> Dict[Tuple, Dict]:
    """Load collective operation configurations for this model.

    Returns:
        Dict mapping (operation_name, dim_param, num_dims) -> config dict
        where config dict contains 'batch_param' and 'model_config_dim'
    """
    conn = sqlite3.connect(db_path)
    query = """
        SELECT co.operation_name, co.dim_param, co.num_dims,
                mcc.batch_param, mcc.model_config_dim
        FROM model_collective_configs mcc
        JOIN coll_ops co ON mcc.collective_id = co.collective_id
        JOIN profiled_models pm ON mcc.model_id = pm.model_id
        WHERE pm.model_name = ?
            AND pm.backend = ?
            AND pm.tp_degree = ?
            AND co.topology = ?
    """

    cursor = conn.execute(query, (model_name, backend, tp_size, topology))
    configs = {}
    for row in cursor:
        op_name, dim_param, num_dims, batch_param, model_config_dim = row
        key = (op_name, dim_param, num_dims)
        configs[key] = {
            'batch_param': batch_param,
            'model_config_dim': model_config_dim
        }
    conn.close()

    if configs:
        print(f"Loaded {len(configs)} collective operation configs")
        for key, config in configs.items():
            op_name, dim_param, num_dims = key
            print(f"  - {op_name} (dim={dim_param}, ndims={num_dims}): "
                    f"{config['batch_param']} × {config['model_config_dim']}")
    else:
        print(f"No collective configs found for {model_name} (tp={tp_size}, topology={topology})")

    return configs



def main():
    """Main test entry point."""

    import argparse
    parser = argparse.ArgumentParser(description="Test DoolySim predictions vs vLLM")
    parser.add_argument('--with-vllm', action='store_true', help='Measure actual vLLM latencies')
    parser.add_argument('--warmup', type=int, default=3, help='Number of warmup iterations')
    parser.add_argument('--tp', type=int, default=1, help='Tensor parallel size for vLLM')
    parser.add_argument('--iterations', type=int, default=10, help='Number of measurement iterations')
    args = parser.parse_args()

    # Configuration
    MODEL = "meta-llama/Llama-3.1-70B"
    CACHE_DIR = "/home/jkim38/dooly/cache/tp4/meta-llama/Llama-3.1-70B/"
    DB_PATH = "/home/jkim38/dooly/profiler/database/dooly_data_new_new.db"
    BACKEND = "FLASHINFER"
    TOPOLOGY = "mew1"
    TP = args.tp
    
    model_collectives=None

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
    model_collectives = load_model_collectives(DB_PATH, BACKEND, MODEL, TOPOLOGY, TP)
    config = load_configs(DB_PATH, MODEL, BACKEND, TP, TOPOLOGY)

    print(f"        Loaded {len(operation_counts) + len(model_collectives)} operation counts")
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
        
        if model_collectives:
            dooly_latency += estimate_comm_latency(
                predictor=predictor,
                configs=config,
                model_collectives=model_collectives,
                batch=batch_config
            )

        print(f"  DoolySim prediction: {dooly_latency:.3f} ms")

        results.append({
            'test_name': test_name,
            'tokens': batch_config['total_tokens'],
            'dooly_ms': dooly_latency,
        })

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"{'Test Name':<35} {'Tokens':>8} {'DoolySim (ms)':>15}")
    print("-" * 80)
    for r in results:
        print(f"{r['test_name']:<35} {r['tokens']:>8} {r['dooly_ms']:>15.3f}")

    print("=" * 80)


if __name__ == "__main__":
    main()
