#!/usr/bin/env python3
"""
Modified version of measure_vllm_streaming.py that doesn't rely on metrics.
Tracks timing manually when disable_log_stats=True.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import List
from tqdm import tqdm
import torch

try:
    from vllm import LLM
    from vllm import SamplingParams
    from vllm.benchmarks.datasets import SampleRequest
    from vllm.benchmarks.lib.endpoint_request_func import RequestFuncOutput
except ImportError:
    print("ERROR: vLLM not available. Install with: pip install vllm")
    sys.exit(1)

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

# Add simulator to path for request generation
simulator_dir = Path(__file__).parent.parent
if str(simulator_dir) not in sys.path:
    sys.path.insert(0, str(simulator_dir))

from vllm_request_generator import generate_vllm_requests_from_instance, generate_sharegpt_requests_from_instance


def run_streaming_benchmark_no_stats(
    llm: LLM,
    requests: List[tuple],  # (request_id, prompt_token_ids, max_tokens, arrival_time)
    warmup: int = 5,
    verbose: bool = False,
) -> tuple[List[RequestFuncOutput], float, dict]:

    engine = llm.llm_engine
    outputs = []
    completed_requests = {}

    # Manual timing tracking (since metrics won't be available)
    request_arrival_times = {}  # req_id -> arrival_time (relative to start)
    request_first_token_times = {}  # req_id -> time when first token generated
    request_completion_times = {}  # req_id -> completion time
    request_token_counts = {}  # req_id -> number of tokens generated
    request_prompt_lens = {}  # req_id -> prompt length

    # Track batch composition per step
    step_batches = []

    # Track per-request step data for ITL calculation
    request_step_data = {}  # req_id -> [{"step_duration": float, "tokens_generated": int}, ...]

    # Sort by arrival time
    requests = sorted(requests, key=lambda x: x[3])

    if verbose:
        print(f"\n{'='*80}")
        print(f"Starting streaming benchmark with {len(requests)} requests")
        print(f"log_stats disabled - using manual timing")
        print(f"{'='*80}\n")

    # warm up
    warmup_sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        ignore_eos=True,
    )

    print(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        from vllm.inputs import TokensPrompt
        prompts = [TokensPrompt(prompt_token_ids=token_ids) for _, token_ids, _, _ in requests]
        llm.generate(prompts, sampling_params=warmup_sampling_params)
        torch.cuda.synchronize()

    print(f"Warm up done. Running real runs.")
    pbar = tqdm(total=len(requests), desc="Processing requests")
    start_time = time.perf_counter()

    request_idx = 0
    step_count = 0

    # Track which requests have generated their first token
    requests_with_first_token = set()

    while request_idx < len(requests) or engine.has_unfinished_requests():
        elapsed_time = time.perf_counter() - start_time

        # Add any requests that have arrived
        while request_idx < len(requests):
            request_id, prompt_token_ids, max_tokens, arrival_time = requests[request_idx]

            if arrival_time <= elapsed_time:
                sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=max_tokens,
                    ignore_eos=True,
                )

                engine.add_request(
                    request_id=request_id,
                    prompt=prompt_token_ids,
                    params=sampling_params,
                )

                # Manual timing tracking
                request_arrival_times[request_id] = elapsed_time
                request_prompt_lens[request_id] = len(prompt_token_ids)
                request_token_counts[request_id] = 0

                if verbose:
                    print(f"[t={elapsed_time:.3f}s] Added request {request_id} "
                          f"(prompt_len={len(prompt_token_ids)}, max_tokens={max_tokens})")

                request_idx += 1
            else:
                break

        # Step the engine if there are requests to process
        if engine.has_unfinished_requests():
            # Access the scheduler through the proper path
            if hasattr(engine, 'engine_core'):
                if hasattr(engine.engine_core, 'engine_core'):
                    scheduler = engine.engine_core.engine_core.scheduler
                else:
                    scheduler = engine.engine_core.scheduler
            else:
                # Fallback for different vLLM versions
                scheduler = engine.scheduler

            # Capture scheduling info
            original_schedule = scheduler.schedule
            captured_sched_output = None

            def capture_schedule():
                nonlocal captured_sched_output
                captured_sched_output = original_schedule()
                return captured_sched_output

            scheduler.schedule = capture_schedule

            # Time the step
            torch.cuda.synchronize()
            step_start = time.perf_counter()
            request_outputs = engine.step()
            torch.cuda.synchronize()
            step_end = time.perf_counter()
            step_duration = step_end - step_start
            current_time = time.perf_counter() - start_time

            scheduler.schedule = original_schedule

            # Extract scheduling metrics
            scheduled_req_ids = []
            total_tokens_scheduled = 0

            if captured_sched_output:
                num_scheduled_new = len(captured_sched_output.scheduled_new_reqs)
                total_tokens_scheduled = captured_sched_output.total_num_scheduled_tokens
                scheduled_req_ids = list(captured_sched_output.num_scheduled_tokens.keys())

                # Track first token generation for new requests
                for req_id_with_hash in scheduled_req_ids:
                    req_id = req_id_with_hash.split('-')[0]  # Remove hash suffix

                    # Check if this is the first time this request generated tokens
                    if req_id not in requests_with_first_token:
                        if req_id in request_arrival_times:  # Ensure it's our request
                            request_first_token_times[req_id] = current_time
                            requests_with_first_token.add(req_id)

                    # Track step data for ITL calculation
                    if req_id not in request_step_data:
                        request_step_data[req_id] = []

                    # Determine if this is prefill or decode
                    is_prefill = False
                    for new_req in captured_sched_output.scheduled_new_reqs:
                        if new_req.req_id == req_id_with_hash:
                            is_prefill = True
                            break

                    request_step_data[req_id].append({
                        "step_duration": step_duration,
                        "tokens_generated": 1 if not is_prefill else 0,  # Only decode generates tokens
                        "is_prefill": is_prefill
                    })

            step_count += 1

            # Record batch composition
            step_batches.append({
                "step_id": step_count,
                "sim_time_ms": elapsed_time * 1000.0,
                "elapsed_time_ms": step_duration * 1000.0,
                "batch_size": len(scheduled_req_ids) if captured_sched_output else 0,
                "total_tokens_scheduled": total_tokens_scheduled if captured_sched_output else 0,
            })

            if verbose and step_count % 10 == 0:
                print(f"[Step {step_count}] time={current_time:.3f}s, "
                      f"duration={step_duration*1000:.2f}ms, "
                      f"tokens={total_tokens_scheduled if captured_sched_output else 0}")

            # Process outputs
            for request_output in request_outputs:
                if request_output.finished:
                    req_id = request_output.request_id

                    # Manual timing and token counting
                    request_completion_times[req_id] = current_time

                    # Count actual generated tokens
                    if hasattr(request_output, 'outputs') and request_output.outputs:
                        request_token_counts[req_id] = len(request_output.outputs[0].token_ids)
                    else:
                        # Default to expected output length if outputs not available
                        request_token_counts[req_id] = 1

                    # Calculate metrics manually
                    arrival_time = request_arrival_times.get(req_id, 0)
                    first_token_time = request_first_token_times.get(req_id, current_time)

                    ttft = first_token_time - arrival_time
                    e2e_latency = current_time - arrival_time

                    # Calculate ITLs from step data
                    itls = []
                    if req_id in request_step_data:
                        for step_data in request_step_data[req_id]:
                            tokens_in_step = step_data["tokens_generated"]
                            for _ in range(tokens_in_step):
                                itls.append(step_data["step_duration"])

                    # If no ITLs calculated, use average
                    if not itls and request_token_counts[req_id] > 1:
                        exit()
                        avg_itl = (e2e_latency - ttft) / (request_token_counts[req_id] - 1)
                        itls = [avg_itl] * (request_token_counts[req_id] - 1)

                    output = RequestFuncOutput(
                        generated_text=request_output.outputs[0].text if hasattr(request_output, 'outputs') and request_output.outputs else "",
                        success=True,
                        latency=e2e_latency,
                        output_tokens=request_token_counts[req_id],
                        ttft=ttft,
                        itl=itls,
                        tpot=0.0,
                        prompt_len=request_prompt_lens[req_id],
                        error="",
                        start_time=arrival_time,
                        input_audio_duration=0.0,
                    )

                    outputs.append(output)
                    completed_requests[req_id] = output

                    pbar.update(1)
                    pbar.set_postfix({
                        'TTFT': f'{ttft*1000:.1f}ms',
                        'tokens': request_token_counts[req_id],
                        'steps': step_count
                    })

                    if verbose:
                        print(f"[t={current_time:.3f}s] Request {req_id} completed "
                              f"(latency={e2e_latency*1000:.1f}ms, "
                              f"ttft={ttft*1000:.1f}ms, "
                              f"tokens={request_token_counts[req_id]})")

    pbar.close()
    end_time = time.perf_counter()
    duration = end_time - start_time

    return outputs, duration, {"batch_composition": step_batches}


def main():
    parser = argparse.ArgumentParser(
        description="Measure vLLM performance with streaming requests (log_stats disabled)"
    )

    # Model configuration
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B",
                       help="Model name")
    parser.add_argument("--backend", type=str, default="FLASHINFER",
                       help="Attention backend (FLASHINFER or FLASH_ATTN)")

    # Request generation
    parser.add_argument("--load-dataset", type=str,
                       help="Load requests from JSONL file (for exact comparison)")
    parser.add_argument("--use-sharegpt", type=str, metavar="SHAREGPT_PATH",
                       help="Use ShareGPT dataset (path to ShareGPT JSON)")
    parser.add_argument("--num-requests", type=int, default=100,
                       help="Number of requests")
    parser.add_argument("--arrival-rate", type=float, default=5.0,
                       help="Request arrival rate (req/s)")
    parser.add_argument("--input-len", type=int, default=1024,
                       help="Mean input length")
    parser.add_argument("--output-len", type=int, default=None,
                       help="Mean output length")
    parser.add_argument("--range-ratio", type=float, default=0.0,
                       help="Length variability (0.0=fixed, 1.0=uniform 0 to 2x)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    # vLLM engine configuration
    parser.add_argument("--max-model-len", type=int, default=16384,
                       help="Maximum model length")
    parser.add_argument("--max-num-seqs", type=int, default=256,
                       help="Maximum number of sequences")
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192,
                       help="Maximum batched tokens")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                       help="GPU memory utilization")
    parser.add_argument("--enable-chunked-prefill", action="store_true",
                       help="Enable chunked prefill")

    # Output options
    parser.add_argument("--output", type=str, default="vllm_streaming_no_stats.json",
                       help="Output JSON file")
    parser.add_argument("--export-requests", type=str,
                       help="Export request dataset to JSONL")
    parser.add_argument("--batch-composition-output", type=str,
                       help="Export batch composition to separate JSON file")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    parser.add_argument("--warmup", type=int, default=3,
                       help="Number of warmup iterations")

    args = parser.parse_args()

    # Auto-generate output filenames based on dataset parameters if not explicitly provided
    if args.use_sharegpt:
        # ShareGPT dataset: sharegpt-{rate}rps-{num_requests}reqs
        dataset_name = f"sharegpt-{args.arrival_rate:.0f}rps-{args.num_requests}reqs"

        # Auto-set export-requests if not provided
        if not args.export_requests:
            args.export_requests = f"./dataset/{dataset_name}.jsonl"

        # Auto-set output files if they're still defaults
        if args.output == "vllm_streaming_no_stats.json":
            args.output = f"./comparison/vllm_no_stats_{dataset_name}.json"

        if not args.batch_composition_output:
            args.batch_composition_output = f"./comparison/vllm_batch_no_stats_{dataset_name}.json"

    elif not args.export_requests and not args.load_dataset:
        # Synthetic dataset: syn-{rate}rps-uniform-{input_len}-{output_len}
        dataset_name = f"syn-{args.arrival_rate:.0f}rps-uniform-{args.input_len}-{args.output_len or 128}"

        # Auto-set export-requests if not provided
        args.export_requests = f"./dataset/{dataset_name}.jsonl"

        # Auto-set output files if they're still defaults
        if args.output == "vllm_streaming_no_stats.json":
            args.output = f"./comparison/vllm_no_stats_{dataset_name}.json"

        if not args.batch_composition_output:
            args.batch_composition_output = f"./comparison/vllm_batch_no_stats_{dataset_name}.json"

    elif args.load_dataset and args.load_dataset.endswith('.jsonl'):
        # Extract dataset name from load-dataset path
        dataset_name = Path(args.load_dataset).stem

        # Auto-set output files if they're still defaults
        if args.output == "vllm_streaming_no_stats.json":
            args.output = f"./comparison/vllm_no_stats_{dataset_name}.json"

        if not args.batch_composition_output:
            args.batch_composition_output = f"./comparison/vllm_batch_no_stats_{dataset_name}.json"

    print("=" * 80)
    print("vLLM STREAMING BENCHMARK (log_stats disabled)")
    print("=" * 80)
    print(f"Model:              {args.model}")
    print(f"Backend:            {args.backend}")
    print(f"Requests:           {args.num_requests}")
    print(f"Arrival rate:       {args.arrival_rate} req/s")
    print(f"Input length:       {args.input_len} (range_ratio={args.range_ratio})")
    print(f"Output length:      {args.output_len} (range_ratio={args.range_ratio})")
    print(f"Max model length:   {args.max_model_len}")
    print(f"Max seqs:           {args.max_num_seqs}")
    print(f"Max batched tokens: {args.max_num_batched_tokens}")
    print(f"Chunked prefill:    {args.enable_chunked_prefill}")
    print(f"Log stats:          DISABLED")
    print("=" * 80)
    print()

    # Create LLM instance with log_stats disabled
    llm = LLM(
        model=args.model,
        attention_backend=args.backend,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        enforce_eager=False,
        load_format="dummy",
        disable_log_stats=True,  # Disable logging
        tensor_parallel_size=1
    )

    # Load or generate requests
    if args.load_dataset:
        # Load from pre-generated JSONL file
        print(f"Loading requests from {args.load_dataset}...")

        tokenizer = llm.get_tokenizer()
        requests = []
        sample_requests = []

        with open(args.load_dataset, 'r') as f:
            for line in f:
                entry = json.loads(line)
                prompt = entry['prompt']
                token_ids = tokenizer.encode(prompt)

                # Streaming request tuple
                requests.append((
                    entry['request_id'],
                    token_ids,
                    entry['output_len'],
                    entry['arrival_time'],
                ))

                # SampleRequest for metrics
                sample_requests.append(SampleRequest(
                    prompt=prompt,
                    prompt_len=entry.get('prompt_len', len(token_ids)),
                    expected_output_len=entry['output_len'],
                ))

        print(f"✓ Loaded {len(requests)} requests from dataset")
    elif args.use_sharegpt:
        # Generate from ShareGPT dataset
        print(f"Generating requests from ShareGPT: {args.use_sharegpt}")
        result = generate_sharegpt_requests_from_instance(
            llm=llm,
            num_requests=args.num_requests,
            custom_output_len=args.output_len,
            arrival_rate=args.arrival_rate,
            sharegpt_path=args.use_sharegpt,
            seed=args.seed,
        )

        # Handle both tuple return (dooly_requests, vllm_requests) and list return
        if isinstance(result, tuple):
            dooly_requests, _ = result  # Ignore vllm_requests
        else:
            dooly_requests = result

        # Export requests if requested
        if args.export_requests:
            print(f"Exporting requests to: {args.export_requests}")
            Path(args.export_requests).parent.mkdir(parents=True, exist_ok=True)
            with open(args.export_requests, 'w') as f:
                for req in dooly_requests:
                    json.dump(req, f)
                    f.write('\n')

        # Convert to request tuples for streaming benchmark
        tokenizer = llm.get_tokenizer()
        requests = []
        sample_requests = []

        for req in dooly_requests:
            token_ids = tokenizer.encode(req['prompt'])
            requests.append((
                req['request_id'],
                token_ids,
                req['output_len'],
                req['arrival_time'],
            ))
            sample_requests.append(SampleRequest(
                prompt=req['prompt'],
                prompt_len=req['prompt_len'],
                expected_output_len=req['output_len'],
            ))
    else:
        # Generate synthetic requests
        print(f"Generating {args.num_requests} synthetic requests...")
        requests = generate_vllm_requests_from_instance(
            llm, args.num_requests, args.arrival_rate,
            args.input_len, args.output_len, args.range_ratio
        )
        sample_requests = None

    # Run benchmark
    print(f"\nRunning benchmark with {len(requests)} requests...")
    outputs, duration, extra_data = run_streaming_benchmark_no_stats(
        llm, requests, warmup=args.warmup, verbose=args.verbose
    )

    outputs_sorted = sorted(outputs, key=lambda x: x.start_time)
    step_based_itls = [output.itl for output in outputs_sorted]

    # Calculate metrics manually (like Dooly does)
    import numpy as np

    ttfts = [output.ttft for output in outputs]
    latencies = [output.latency for output in outputs]
    input_lens = [output.prompt_len for output in outputs]
    output_lens = [output.output_tokens for output in outputs]

    # Calculate throughput
    total_output_tokens = sum(output_lens)
    total_input_tokens = sum(input_lens)
    request_throughput = len(outputs) / duration
    output_throughput = total_output_tokens / duration
    total_throughput = (total_input_tokens + total_output_tokens) / duration

    # Calculate TTFT percentiles
    ttfts_ms = np.array(ttfts) * 1000  # Convert to ms
    ttft_mean = np.mean(ttfts_ms)
    ttft_median = np.median(ttfts_ms)
    ttft_p95 = np.percentile(ttfts_ms, 95)
    ttft_p99 = np.percentile(ttfts_ms, 99)

    # Calculate ITL stats
    all_itls = []
    for output in outputs:
        if output.itl:
            all_itls.extend(output.itl)

    if all_itls:
        itls_ms = np.array(all_itls) * 1000  # Convert to ms
        itl_mean = np.mean(itls_ms)
        itl_median = np.median(itls_ms)
        itl_p95 = np.percentile(itls_ms, 95)
        itl_p99 = np.percentile(itls_ms, 99)
    else:
        itl_mean = itl_median = itl_p95 = itl_p99 = 0.0

    # Prepare results (matching Dooly format)
    results = {
        "endpoint_type": "vllm",
        "backend": args.backend,
        "model": args.model,
        "num_prompts": len(requests),
        "arrival_rate": args.arrival_rate,
        "duration": duration,
        "request_throughput": request_throughput,
        "output_throughput": output_throughput,
        "total_throughput": total_throughput,
        
        "mean_ttft_ms": ttft_mean,
        "median_ttft_ms": ttft_median,
        "p95_ttft_ms": ttft_p95,
        "p99_ttft_ms": ttft_p99,
        "mean_itl_ms": itl_mean,
        "median_itl_ms": itl_median,
        "p95_itl_ms": itl_p95,
        "p99_itl_ms": itl_p99,
        
        "ttfts": ttfts,  # In seconds
        "itls": step_based_itls,  # In seconds
        "latencies": latencies,  # In seconds
        "input_lens": input_lens,
        "output_lens": output_lens,
    }

    # Save results
    print(f"\nSaving results to: {args.output}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved")

    # Save batch composition
    if args.batch_composition_output and "batch_composition" in extra_data:
        print(f"Saving batch composition to: {args.batch_composition_output}")
        Path(args.batch_composition_output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.batch_composition_output, 'w') as f:
            json.dump(extra_data["batch_composition"], f, indent=2)
        print(f"✓ Batch composition saved")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Benchmark Summary (log_stats disabled)")
    print(f"{'='*60}")
    print(f"Duration: {duration:.2f}s")
    print(f"Request Throughput: {request_throughput:.2f} req/s")
    print(f"Output Throughput: {output_throughput:.2f} tok/s")
    print(f"Total Throughput: {total_throughput:.2f} tok/s")
    print(f"Mean TTFT: {ttft_mean:.2f}ms")
    print(f"Median TTFT: {ttft_median:.2f}ms")
    print(f"P95 TTFT: {ttft_p95:.2f}ms")
    print(f"P99 TTFT: {ttft_p99:.2f}ms")
    print(f"Mean ITL: {itl_mean:.2f}ms")
    print(f"P99 ITL: {itl_p99:.2f}ms")


if __name__ == "__main__":
    main()