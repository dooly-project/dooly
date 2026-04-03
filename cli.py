"""Command-line interface for DoolySim."""

import argparse
from .core.dooly_sim import DoolySim
from .request import RequestStream
from .analysis.reporters import print_summary

# Global verbosity flag - can be imported by other modules
VERBOSE = False 


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="DoolySim - vLLM Scheduler Simulator")

    # Model configuration
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B",
                       help="Model name to simulate")
    parser.add_argument("--backend", type=str, default="FLASHINFER",
                       help="Backend used for profiling")

    # Paths
    parser.add_argument("--cache-dir", type=str,
                       default="/home/jkim38/dooly/simulator/core/cache/llama-3.1-8b-FLASHINFER",
                       help="Directory containing cached regression models")
    parser.add_argument("--db-path", type=str,
                       default="/home/jkim38/dooly/profiler/database/dooly_data.db",
                       help="Path to profiling database")

    # vLLM configuration
    parser.add_argument("--max-num-seqs", type=int, default=256,
                       help="Maximum number of sequences")
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192,
                       help="Maximum number of batched tokens")
    parser.add_argument("--max-model-len", type=int, default=8192,
                       help="Maximum model context length")
    parser.add_argument("--gpu-mem-util", type=float, default=0.9,
                       help="GPU memory utilization")

    # Synthetic Request Generation Options 
    # (might be better to move this to some distinct module for request generation)
    parser.add_argument("--num-requests", type=int, default=20,
                       help="Number of requests to simulate")
    parser.add_argument("--arrival-rate", type=float, default=5.0,
                       help="Request arrival rate (requests/sec)")
    parser.add_argument("--prompt-len-min", type=int, default=256,
                       help="Minimum prompt length")
    parser.add_argument("--prompt-len-max", type=int, default=4096,
                       help="Maximum prompt length")
    parser.add_argument("--output-len-min", type=int, default=64,
                       help="Minimum output length")
    parser.add_argument("--output-len-max", type=int, default=512,
                       help="Maximum output length")
    
    # Simulator Operations
    parser.add_argument("--max-steps", type=int, default=100000,
                       help="Maximum simulation steps")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--show-per-request", action="store_true",
                       help="Show detailed per-request statistics")

    # CSV export options
    parser.add_argument("--export-csv", action="store_true",
                       help="Export CSV files for analysis")
    parser.add_argument("--csv-output-dir", type=str, default="./csv_output",
                       help="Directory to save CSV files")
    parser.add_argument("--csv-prefix", type=str, default="simulation",
                       help="Prefix for CSV filenames")

    # Fallback
    parser.add_argument("--fallback-duration-ms", type=float, default=10.0,
                       help="Fallback step duration in ms if models fail to load")

    args = parser.parse_args()

    # Set global VERBOSE flag
    global VERBOSE
    VERBOSE = args.verbose

    # Print header
    print()
    print("=" * 60)
    print("DoolySim (w Emulated vLLM scheduler)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Backend: {args.backend}")
    print(f"Cache dir: {args.cache_dir}")
    print(f"DB path: {args.db_path}")
    print()

    # Create request stream
    stream = RequestStream(
        num_requests=args.num_requests,
        arrival_rate=args.arrival_rate,
        prompt_len_range=(args.prompt_len_min, args.prompt_len_max),
        output_len_range=(args.output_len_min, args.output_len_max),
        seed=args.seed,
    )

    # Create simulator
    simulator = DoolySim(
        step_duration_ms=args.fallback_duration_ms,
        model=args.model,
        args={
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_mem_util,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "enable_prefix_caching": False,
        },
        cache_dir=args.cache_dir,
        db_path=args.db_path,
        backend=args.backend,
    )

    simulator.dump_config()

    # Run simulation
    result = simulator.run(
        request_stream=stream.generate(),
        max_steps=args.max_steps,
        verbose=args.verbose,
    )

    # Print results
    print_summary(result, show_per_request=args.show_per_request)

    # Export CSV files if requested
    if args.export_csv:
        result.export_all_csvs(args.csv_output_dir, args.csv_prefix)


if __name__ == "__main__":
    main()
