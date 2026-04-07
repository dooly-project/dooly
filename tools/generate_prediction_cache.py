#!/usr/bin/env python3
"""
Generate CSV prediction cache files from profiled data.

This script trains the regression models and generates CSV files containing
all predictions for fast lookup during simulation.

Usage:
    python -m simulator.tools.generate_prediction_cache --model meta-llama/Llama-3.1-8B --backend FLASHINFER
"""

import argparse
import sys
from pathlib import Path

# Add simulator to path
simulator_dir = Path(__file__).parent.parent
if str(simulator_dir) not in sys.path:
    sys.path.insert(0, str(simulator_dir))

from simulator.estimator import EstimatorConfig, SimplifiedExecutionTimeEstimator
from simulator.core.prediction_cache import generate_prediction_cache_from_estimator


def main():
    parser = argparse.ArgumentParser(
        description="Generate CSV prediction cache from profiled data"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., 'meta-llama/Llama-3.1-8B')"
    )
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        help="Backend type (e.g., 'FLASHINFER' or 'FLASH_ATTN')"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="/home/jkim38/dooly/profiler/database/dooly_data.db",
        help="Path to profiling database"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for predictions (auto-generated if not specified)"
    )
    parser.add_argument(
        "--model-cache-dir",
        type=str,
        default="/home/jkim38/dooly/simulator/cache",
        help="Directory for caching trained sklearn models"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if cache exists"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Maximum number of tokens for predictions"
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=16,
        help="Maximum batch size for decode predictions"
    )
    parser.add_argument(
        "--max-prefill-chunk",
        type=int,
        default=16384,
        help="Maximum prefill chunk size for predictions"
    )
    parser.add_argument(
        "--kv-cache-granularity",
        type=int,
        default=64,
        help="Granularity for KV cache size predictions"
    )

    args = parser.parse_args()

    # Generate cache directory name if not specified
    if args.cache_dir is None:
        model_safe = args.model.replace('/', '-').lower()
        backend_safe = args.backend.lower()
        args.cache_dir = f"/home/jkim38/dooly/simulator/core/cache/llama-3.1-8b-FLASHINFER"

    print("\n" + "=" * 80)
    print("PREDICTION CACHE GENERATOR")
    print("=" * 80)
    print(f"Model:               {args.model}")
    print(f"Backend:             {args.backend}")
    print(f"Database:            {args.db_path}")
    print(f"Prediction cache:    {args.cache_dir}")
    print(f"Model cache:         {args.model_cache_dir}")
    print(f"Max tokens:          {args.max_tokens}")
    print(f"Max batch size:      {args.max_batch_size}")
    print(f"Max prefill chunk:   {args.max_prefill_chunk}")
    print(f"KV cache granularity: {args.kv_cache_granularity}")
    print(f"Force regenerate:    {args.force}")
    print("=" * 80)

    # Step 1: Train or load regression models
    print("\n" + "=" * 80)
    print("STEP 1: TRAIN OR LOAD REGRESSION MODELS")
    print("=" * 80)

    config = EstimatorConfig(
        db_path=args.db_path,
        cache_dir=args.model_cache_dir,
        model_name=args.model,
        backend=args.backend,
        no_cache=False,
        prediction_max_tokens=args.max_tokens,
        prediction_max_batch_size=args.max_batch_size,
        prediction_max_prefill_chunk_size=args.max_prefill_chunk,
        kv_cache_granularity=args.kv_cache_granularity,
    )

    estimator = SimplifiedExecutionTimeEstimator(config)

    # Train all models (will load from cache if available)
    estimator.train_all_models()

    # Step 2: Generate CSV prediction cache
    print("\n" + "=" * 80)
    print("STEP 2: GENERATE CSV PREDICTION CACHE")
    print("=" * 80)

    generate_prediction_cache_from_estimator(
        estimator=estimator,
        cache_dir=args.cache_dir,
        force_regenerate=args.force
    )

    # Step 3: Validate predictions
    print("\n" + "=" * 80)
    print("STEP 3: VALIDATE PREDICTIONS")
    print("=" * 80)

    estimator.validate_predictions(num_samples=5)

    # Print summary
    print("\n" + "=" * 80)
    print("CACHE GENERATION COMPLETE")
    print("=" * 80)
    print(f"Generated {len(estimator.models)} operation prediction caches")
    print(f"Cache directory: {args.cache_dir}")
    print("\nTo use this cache in your simulator:")
    print(f"  cache_dir = '{args.cache_dir}'")
    print(f"  model_name = '{args.model}'")
    print(f"  backend = '{args.backend}'")
    print("=" * 80)


if __name__ == "__main__":
    main()
