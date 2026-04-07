"""Command-line interface for DoolySim."""

import argparse
import os
import json
from .core.dooly_sim import DoolySim
from .request import RequestStream, RequestConfig
from .analysis.reporters import print_summary

# Global verbosity flag - can be imported by other modules
VERBOSE = False


def load_requests_from_jsonl(jsonl_path: str, tokenizer):
    """Load requests from JSONL file.

    Args:
        jsonl_path: Path to JSONL file with format:
            {"prompt": "...", "output_len": N, "request_id": "...", "arrival_time": T}
        tokenizer: Tokenizer instance from simulator

    Returns:
        List of RequestConfig objects
    """
    print(f"Loading requests from {jsonl_path}...")

    requests = []
    with open(jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            entry = json.loads(line)

            # Tokenize prompt
            token_ids = tokenizer.encode(entry['prompt'])

            # Create RequestConfig
            req = RequestConfig(
                request_id=entry.get('request_id', f"req_{line_num-1}"),
                prompt_token_ids=token_ids,
                max_tokens=entry['output_len'],
                arrival_time=entry['arrival_time'],
            )
            requests.append(req)

    print(f"✓ Loaded {len(requests)} requests from dataset")
    if len(requests) > 0:
        print(f"  Sample prompt lengths: {[len(r.prompt_token_ids) for r in requests[:5]]}")
        print(f"  Sample output lengths: {[r.max_tokens for r in requests[:5]]}")
        print(f"  Sample arrival times: {[f'{r.arrival_time:.3f}s' for r in requests[:5]]}")

    return requests 


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
                       default="/home/jkim38/dooly/simulator/core/cache/llama-3.1-8b-FLASHINFER/csv_predictions",
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

    # Request Generation Options
    parser.add_argument("--use-vllm-dataset", action="store_true",
                       help="Use vLLM's RandomDataset generator for reproducible benchmarks")
    parser.add_argument("--use-sharegpt", type=str, metavar="SHAREGPT_PATH",
                       help="Use ShareGPT dataset (path to ShareGPT JSON file)")
    parser.add_argument("--vllm-dataset-output", type=str,
                       help="Export vLLM dataset to JSONL file for vllm bench serve")
    parser.add_argument("--load-dataset", type=str,
                       help="Load requests from a pre-generated JSONL file (for exact comparison)")

    # Common parameters for both generators
    parser.add_argument("--num-requests", type=int, default=20,
                       help="Number of requests to simulate")
    parser.add_argument("--arrival-rate", type=float, default=5.0,
                       help="Request arrival rate (requests/sec)")

    # DoolySim's RequestStream parameters (range-based)
    parser.add_argument("--prompt-len-min", type=int, default=256,
                       help="[RequestStream] Minimum prompt length")
    parser.add_argument("--prompt-len-max", type=int, default=4096,
                       help="[RequestStream] Maximum prompt length")
    parser.add_argument("--output-len-min", type=int, default=64,
                       help="[RequestStream] Minimum output length")
    parser.add_argument("--output-len-max", type=int, default=512,
                       help="[RequestStream] Maximum output length")

    # vLLM's RandomDataset parameters (mean + range_ratio)
    parser.add_argument("--input-len", type=int, default=1024,
                       help="[vLLM] Mean input length")
    parser.add_argument("--output-len", type=int, default=128,
                       help="[vLLM] Mean output length")
    parser.add_argument("--range-ratio", type=float, default=0.0,
                       help="[vLLM] Length variability (0.0=fixed, 1.0=uniform 0 to 2x mean)")
    
    # Simulator Operations
    parser.add_argument("--max-steps", type=int, default=100000,
                       help="Maximum simulation steps")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--show-per-request", action="store_true",
                       help="Show detailed per-request statistics")
    parser.add_argument("--use-wall-clock-time", action="store_true",
                       help="Use wall-clock time for request arrivals (for fair comparison with vLLM)")

    # Export options
    parser.add_argument("--export-csv", action="store_true",
                       help="Export CSV files for analysis")
    parser.add_argument("--csv-output-dir", type=str, default="./csv_output",
                       help="Directory to save CSV files")
    parser.add_argument("--csv-prefix", type=str, default="simulation",
                       help="Prefix for CSV filenames")
    parser.add_argument("--export-json", action="store_true",
                       help="Export JSON in vLLM benchmark format")
    parser.add_argument("--json-output", type=str,
                       help="Output path for vLLM-format JSON file")

    # Fallback
    parser.add_argument("--fallback-duration-ms", type=float, default=0.0,
                       help="Fallback step duration in ms if models fail to load")
    parser.add_argument("--cpu-overhead-ms", type=float, default=10.0)

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

    # Create simulator first (so we have tokenizer for loading dataset)
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
        cpu_overhead=args.cpu_overhead_ms,
    )

    simulator.dump_config()

    # Generate or load requests
    if args.load_dataset:
        # Load from pre-generated JSONL file
        print(f"Loading dataset from {args.load_dataset}")
        dooly_requests = load_requests_from_jsonl(args.load_dataset, simulator.tokenizer)
        requests = iter(dooly_requests)
    elif args.use_sharegpt:
        # Generate using ShareGPT dataset
        print(f"Using ShareGPT dataset for request generation: {args.use_sharegpt}")
        from .vllm_request_generator import generate_sharegpt_requests_from_instance, export_vllm_dataset_jsonl

        dooly_requests, vllm_requests = generate_sharegpt_requests_from_instance(
            llm=simulator.llm,
            num_requests=args.num_requests,
            arrival_rate=args.arrival_rate,
            sharegpt_path=args.use_sharegpt,
            seed=args.seed,
        )

        # Export dataset if requested
        if args.vllm_dataset_output:
            arrival_times = [r.arrival_time for r in dooly_requests]
            export_vllm_dataset_jsonl(vllm_requests, arrival_times, args.vllm_dataset_output)

        requests = iter(dooly_requests)
    elif args.use_vllm_dataset:
        # Generate using vLLM RandomDataset
        print("Using vLLM RandomDataset for request generation")
        from .vllm_request_generator import generate_vllm_requests, export_vllm_dataset_jsonl

        dooly_requests, vllm_requests = generate_vllm_requests(
            model=args.model,
            num_requests=args.num_requests,
            arrival_rate=args.arrival_rate,
            input_len=args.input_len,
            output_len=args.output_len,
            range_ratio=args.range_ratio,
            seed=args.seed,
        )

        # Export vLLM dataset if requested
        if args.vllm_dataset_output:
            arrival_times = [r.arrival_time for r in dooly_requests]
            export_vllm_dataset_jsonl(vllm_requests, arrival_times, args.vllm_dataset_output)

        requests = iter(dooly_requests)
    else:
        # Use DoolySim's own RequestStream
        print("Using DoolySim RequestStream for request generation")
        stream = RequestStream(
            num_requests=args.num_requests,
            arrival_rate=args.arrival_rate,
            prompt_len_range=(args.prompt_len_min, args.prompt_len_max),
            output_len_range=(args.output_len_min, args.output_len_max),
            seed=args.seed,
        )
        requests = stream.generate()

    # Run simulation
    result = simulator.run(
        request_stream=requests,
        max_steps=args.max_steps,
        verbose=args.verbose,
        use_simulated_time=not args.use_wall_clock_time,  # Invert flag
    )

    # Print results
    print_summary(result, show_per_request=args.show_per_request)

    # Export CSV files if requested
    if args.export_csv:
        result.export_all_csvs(args.csv_output_dir, args.csv_prefix)

    # Export JSON in vLLM format if requested
    if args.export_json or args.json_output:
        json_path = args.json_output if args.json_output else os.path.join(args.csv_output_dir, f"{args.csv_prefix}.json")
        result.export_vllm_json(json_path, args.model, args.arrival_rate)


if __name__ == "__main__":
    main()
