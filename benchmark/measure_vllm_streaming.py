#!/usr/bin/env python3
"""
Measure vLLM performance with streaming requests (arrival times).

This script uses vLLM's LLMEngine directly to add requests over time,
mimicking the online serving behavior that dooly_sim.py simulates.

This provides a fair baseline comparison where both systems:
1. Use the same request stream (from RandomDataset)
2. Feed requests at the same arrival times
3. Use the same vLLM scheduler logic
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import torch

try:
    from vllm import LLM 
    from vllm import EngineArgs, LLMEngine, SamplingParams
    from vllm.benchmarks.serve import calculate_metrics, BenchmarkMetrics
    from vllm.benchmarks.datasets import SampleRequest
    from vllm.benchmarks.lib.endpoint_request_func import RequestFuncOutput
except ImportError:
    print("ERROR: vLLM not available. Install with: pip install vllm")
    sys.exit(1)

import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  

# Add simulator to path for request generation
simulator_dir = Path(__file__).parent.parent
if str(simulator_dir) not in sys.path:
    sys.path.insert(0, str(simulator_dir))

from vllm_request_generator import generate_vllm_requests_from_instance, generate_sharegpt_requests_from_instance


def run_streaming_benchmark(
    llm: LLM,
    requests: List[tuple],  # (request_id, prompt_token_ids, max_tokens, arrival_time)
    warmup: int = 5,
    verbose: bool = False,
) -> tuple[List[RequestFuncOutput], float, dict]:

    engine = llm.llm_engine
    outputs = []
    completed_requests = {}

    # Track batch information for each request
    request_batch_info = {}  # req_id -> {"ttft_batch_size": N, "ttft_step": M, "first_token_generated": False}

    # Track batch composition per step
    step_batches = []  # List of {"step": N, "batch_size": M, "req_ids": [...]}

    # Track first arrival time to normalize all times (like measure_vllm_direct.py does on line 142)
    first_request_arrival_time = None

    # Sort by arrival time
    requests = sorted(requests, key=lambda x: x[3])

    if verbose:
        print(f"\n{'='*80}")
        print(f"Starting streaming benchmark with {len(requests)} requests")
        print(f"{'='*80}\n")

    # warm up
    warmup_sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        ignore_eos=True,
    )

    print(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        # Pass token IDs as TextTokensPrompt objects
        from vllm.inputs import TokensPrompt
        prompts = [TokensPrompt(prompt_token_ids=token_ids) for _, token_ids, _, _ in requests]
        llm.generate(prompts, sampling_params=warmup_sampling_params)
        torch.cuda.synchronize()

    print(f"Warm up done. Running real runs.")
    # tqdm
    pbar = tqdm(total=len(requests), desc="Processing requests")
    start_time = time.perf_counter()

    # Record wall-clock time at benchmark start (for converting our relative times to wall-clock)
    benchmark_start_wall_clock = time.time()

    # Map request_id to its arrival_time from requests list (relative to benchmark start)
    request_arrival_times = {}

    # Track time base offset (wall-clock - engine_time), set from first completed request
    time_base_offset = None

    request_idx = 0
    step_count = 0
    while request_idx < len(requests) or engine.has_unfinished_requests():
        # Calculate elapsed wall-clock time since benchmark start
        # This ensures requests are fed at the same absolute rate regardless of execution speed
        elapsed_time = time.perf_counter() - start_time

        # Add any requests that have arrived based on wall-clock time
        while request_idx < len(requests):
            request_id, prompt_token_ids, max_tokens, arrival_time = requests[request_idx]

            if arrival_time <= elapsed_time:
                # Create sampling params
                sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=max_tokens,
                    ignore_eos=True,
                )
                print(f"req_id: {request_id} - max_tokens: {max_tokens}")

                # Add request to engine
                engine.add_request(
                    request_id=request_id,
                    prompt=prompt_token_ids,
                    params=sampling_params,
                )

                # Store the arrival_time from our requests list
                request_arrival_times[request_id] = arrival_time

                if verbose:
                    print(f"[t={elapsed_time:.3f}s] Added request {request_id} "
                          f"(prompt_len={len(prompt_token_ids)}, max_tokens={max_tokens})")

                request_idx += 1
            else:
                break

        # Step the engine if there are requests to process
        if engine.has_unfinished_requests():
            step_start = time.perf_counter()
            request_outputs = engine.step()
            step_end = time.perf_counter()
            step_duration = step_end - step_start

            step_count += 1

            # Log scheduling decision
            scheduler = engine.engine_core.engine_core.scheduler  # Get first scheduler (for TP=1)
            running_reqs = {req.request_id for req in scheduler.running}
            num_tokens_dict = {req.request_id: req.num_computed_tokens for req in scheduler.running}
            total_tokens = sum(num_tokens_dict.values())
            current_batch_size = len(running_reqs)

            # Record batch composition for this step
            step_batches.append({
                "step": step_count,
                "batch_size": current_batch_size,
                "req_ids": sorted(running_reqs),
                "total_tokens": total_tokens
            })

            if verbose:
                print(f"[Step {step_count}] SCHED: reqs={sorted(running_reqs)}, "
                      f"tokens={total_tokens}, tokens_per_req={dict(sorted(num_tokens_dict.items()))}")

            # Process outputs
            for request_output in request_outputs:
                # if verbose:
                #     print(f"Step: {step_count}, request id: {request_output.request_id}, finished: {request_output.finished}")

                if request_output.finished:
                    # Request completed
                    req_id = request_output.request_id
                    
                    # Skip first request for debugging (can be outlier due to initialization)
                    # if req_id == "req_0":
                    #     continue  

                    # Get metrics from request_output
                    metrics = request_output.metrics

                    # Calculate ITLs
                    itls = []
                    if metrics.num_generation_tokens > 1:
                        avg_itl = ((metrics.last_token_ts - metrics.first_token_ts) / metrics.num_generation_tokens)
                        itls = [avg_itl] * metrics.num_generation_tokens

                    # Use arrival_time from our requests list (ground truth)
                    relative_start_time = request_arrival_times[req_id]

                    # Debug: print metrics values for first request to check available fields
                    if len(outputs) == 0 and verbose:
                        print(f"\n[DEBUG] First request metrics (checking for scheduled_ts):")
                        print(f"  arrival_time: {metrics.arrival_time}")
                        print(f"  queued_ts: {getattr(metrics, 'queued_ts', 'N/A')}")
                        print(f"  scheduled_ts: {getattr(metrics, 'scheduled_ts', 'N/A')}")
                        print(f"  first_token_ts: {metrics.first_token_ts}")
                        print(f"  last_token_ts: {metrics.last_token_ts}")
                        print(f"  first_token_latency: {metrics.first_token_latency}")
                        print(f"  num_generation_tokens: {metrics.num_generation_tokens}")
                        print(f"  relative_start_time: {relative_start_time}\n")

                    if req_id == "req_99":
                        print(f"\n[DEBUG] req_99 finishing:")
                        print(f"  metrics.num_generation_tokens: {metrics.num_generation_tokens}")
                        print(f"  len(request_output.outputs[0].token_ids): {len(request_output.outputs[0].token_ids) if request_output.outputs else 'N/A'}")
                        print(f"  prompt_len: {len(request_output.prompt_token_ids)}")
                        # exit()

                    # Set time base offset from first completed request
                    # Offset = wall_clock_time - engine_time (constant for all requests)
                    if time_base_offset is None:
                        time_base_offset = metrics.arrival_time - metrics.queued_ts

                    # For THIS request, convert its actual arrival time (from vLLM metrics) to engine time
                    # metrics.arrival_time = wall-clock when add_request() was called for THIS request
                    engine_time_arrival = metrics.arrival_time - time_base_offset

                    # TTFT = time from when THIS request was added to first token (includes queue + prefill)
                    ttft_with_queue = metrics.first_token_ts - engine_time_arrival

                    # E2E latency = time from when THIS request was added to last token
                    e2e_latency = metrics.last_token_ts - engine_time_arrival

                    # Debug first completed request to check if timestamps are in same time base
                    if len(outputs) == 0:
                        # Test if vLLM uses same time base for all timestamps (as docs claim)
                        direct_ttft = metrics.first_token_ts - metrics.arrival_time

                        print(f"\n[DEBUG] First completed request ({req_id}) - Testing time base:")
                        print(f"  metrics.arrival_time: {metrics.arrival_time:.6f}")
                        print(f"  metrics.first_token_ts: {metrics.first_token_ts:.6f}")
                        print(f"  ")
                        print(f"  Direct subtraction: {direct_ttft:.6f}s = {direct_ttft*1000:.2f}ms")
                        print(f"  (If negative/huge, timestamps are in different time bases)")
                        print(f"  ")
                        print(f"  Current method (with conversion):")
                        print(f"    time_base_offset: {time_base_offset:.6f}")
                        print(f"    TTFT (converted): {ttft_with_queue:.6f}s = {ttft_with_queue*1000:.2f}ms")
                        print(f"  ")
                        print(f"  metrics.first_token_latency: {metrics.first_token_latency:.6f}s = {metrics.first_token_latency*1000:.2f}ms")

                    # Also debug req_0 specifically
                    if req_id == "req_0":
                        print(f"\n[DEBUG] req_0 TTFT calculation:")
                        print(f"  metrics.arrival_time: {metrics.arrival_time:.6f}")
                        print(f"  engine_time_arrival: {engine_time_arrival:.6f}")
                        print(f"  first_token_ts: {metrics.first_token_ts:.6f}")
                        print(f"  TTFT: {ttft_with_queue:.6f}s = {ttft_with_queue*1000:.2f}ms")
                        print(f"  E2E: {e2e_latency:.6f}s")

                    output = RequestFuncOutput(
                        generated_text=request_output.outputs[0].text if request_output.outputs else "",
                        success=True,
                        latency=e2e_latency,  # E2E latency from arrival to completion
                        output_tokens=metrics.num_generation_tokens,
                        ttft=ttft_with_queue,  # Include queue time for fair comparison
                        itl=itls,
                        tpot=0.0,  # Will be calculated by calculate_metrics
                        prompt_len=len(request_output.prompt_token_ids),
                        error="",
                        start_time=relative_start_time,  # Use relative time, not absolute
                        input_audio_duration=0.0,
                    )

                    if req_id == "req_99":
                        print(f"  [IMMEDIATELY AFTER CREATION] output.output_tokens: {output.output_tokens}")
                        print(f"  [IMMEDIATELY AFTER CREATION] output.prompt_len: {output.prompt_len}")

                    outputs.append(output)
                    completed_requests[req_id] = output

                    if req_id == "req_99":
                        print(f"  [AFTER APPEND] outputs list has {len(outputs)} items")
                        print(f"  [AFTER APPEND] outputs[-1].output_tokens: {outputs[-1].output_tokens}")
                        print(f"  [AFTER APPEND] outputs[-1].prompt_len: {outputs[-1].prompt_len}")

                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({
                        'TTFT': f'{output.ttft*1000:.1f}ms',
                        'tokens': output.output_tokens,
                        'steps': step_count
                    })

                    if verbose:
                        current_time = time.perf_counter() - start_time
                        itl_str = f"itl={output.itl[0]*1000:.1f}ms" if output.itl else "itl=N/A"
                        print(f"[t={current_time:.3f}s] Request {req_id} completed "
                              f"(latency={output.latency*1000:.1f}ms, "
                              f"ttft={output.ttft*1000:.1f}ms, "
                              f"{itl_str}, "
                              f"tokens={output.output_tokens})")

    # Close progress bar
    pbar.close()

    end_time = time.perf_counter()
    duration = end_time - start_time

    if verbose:
        print(f"\n{'='*80}")
        print(f"Benchmark completed:")
        print(f"  Total steps: {step_count}")
        print(f"  Wall-clock time: {duration:.2f}s")
        print(f"  Completed requests: {len(outputs)}/{len(requests)}")
        print(f"{'='*80}\n")

    return outputs, duration, step_batches

def main():
    parser = argparse.ArgumentParser(
        description="Measure vLLM performance with streaming requests"
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
    parser.add_argument("--max-model-len", type=int, default=8192,
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
    parser.add_argument("--output", type=str, default="vllm_streaming.json",
                       help="Output JSON file")
    parser.add_argument("--export-requests", type=str,
                       help="Export request dataset to JSONL")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")

    args = parser.parse_args()

    print("=" * 80)
    print("vLLM STREAMING BENCHMARK (Arrival Time Simulation)")
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
    print("=" * 80)
    print()
    
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
            disable_log_stats=False,
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
        dooly_requests, vllm_requests = generate_sharegpt_requests_from_instance(
            llm=llm,
            num_requests=args.num_requests,
            custom_output_len=args.output_len,
            arrival_rate=args.arrival_rate,
            sharegpt_path=args.use_sharegpt,
            seed=args.seed,
        )

        # Export requests if requested
        if args.export_requests:
            from vllm_request_generator import export_vllm_dataset_jsonl
            arrival_times = [r.arrival_time for r in dooly_requests]
            export_vllm_dataset_jsonl(vllm_requests, arrival_times, args.export_requests)

        # Convert to format for streaming benchmark
        requests = [
            (req.request_id, req.prompt_token_ids, req.max_tokens, req.arrival_time)
            for req in dooly_requests
        ]

        sample_requests = [
            SampleRequest(
                prompt=vllm_req.prompt,
                prompt_len=vllm_req.prompt_len,
                expected_output_len=vllm_req.expected_output_len,
            )
            for vllm_req in vllm_requests
        ]
        
        print(f"[REQUESTS] Req_id: {requests[-1][0]} - output len: {requests[-1][2]}")
        print(f"[VLLM_REQUESTS] Req_id: {vllm_requests[-1].request_id} - output len: {vllm_requests[-1].expected_output_len}")
        # exit(0)
    else:
        # Generate requests using RandomDataset (same as dooly_sim)
        print("Generating requests with vLLM RandomDataset...")
        dooly_requests, vllm_requests = generate_vllm_requests_from_instance(
            llm=llm,
            num_requests=args.num_requests,
            arrival_rate=args.arrival_rate,
            input_len=args.input_len,
            output_len=args.output_len,
            range_ratio=args.range_ratio,
            seed=args.seed,
        )

        # Export requests if requested
        if args.export_requests:
            from vllm_request_generator import export_vllm_dataset_jsonl
            arrival_times = [r.arrival_time for r in dooly_requests]
            export_vllm_dataset_jsonl(vllm_requests, arrival_times, args.export_requests)

        # Convert to format for streaming benchmark
        requests = [
            (req.request_id, req.prompt_token_ids, req.max_tokens, req.arrival_time)
            for req in dooly_requests
        ]

        sample_requests = [
            SampleRequest(
                prompt=vllm_req.prompt,
                prompt_len=vllm_req.prompt_len,
                expected_output_len=vllm_req.expected_output_len,
            )
            for vllm_req in vllm_requests
        ]
    

    print(f"\n✓ Ready with {len(requests)} requests")
    print(f"  First arrival: {requests[0][3]:.3f}s")
    print(f"  Last arrival:  {requests[-1][3]:.3f}s")
    print()

    # Run streaming benchmark
    print("Running streaming benchmark...")
    outputs, duration, step_batches = run_streaming_benchmark(
        llm=llm,
        warmup=3,
        requests=requests,
        verbose=args.verbose,
    )

    # Sort outputs by start_time to match input_requests order
    # CRITICAL: calculate_metrics assumes outputs[i] corresponds to input_requests[i]
    # But outputs list is in completion order! Must sort by arrival time (start_time)
    outputs_sorted = sorted(outputs, key=lambda x: x.start_time)

    # print(f"\n[DEBUG] Verifying index alignment after sorting:")
    # print(f"  outputs_sorted[99]: start={outputs_sorted[99].start_time:.3f}s, tokens={outputs_sorted[99].output_tokens}, prompt={outputs_sorted[99].prompt_len}")
    # print(f"  sample_requests[99]: prompt={sample_requests[99].prompt_len}, expected_output={sample_requests[99].expected_output_len}")

    print(f"\n[DEBUG] Checking timestamp ranges:")
    print(f"  Min start_time: {min(o.start_time for o in outputs_sorted):.6f}s")
    print(f"  Max start_time: {max(o.start_time for o in outputs_sorted):.6f}s")
    print(f"  Sample latencies (first 5): {[f'{o.latency:.3f}s' for o in outputs_sorted[:5]]}")
    print(f"  Sample TTFTs (first 5): {[f'{o.ttft:.3f}s' for o in outputs_sorted[:5]]}")

    # Calculate metrics using vLLM's standard function
    print("\nCalculating metrics...")
    bench_metrics, actual_output_lens = calculate_metrics(
        input_requests=sample_requests,
        outputs=outputs_sorted,
        dur_s=duration,
        tokenizer=None,
        selected_percentiles=[50.0, 95.0, 99.0],
        goodput_config_dict={},
    )

    print(f"\n[DEBUG] actual_output_lens from calculate_metrics: {actual_output_lens[-5:]}")
    print(f"[DEBUG] output.output_tokens from outputs list: {[o.output_tokens for o in outputs[-5:]]}")

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Completed:             {bench_metrics.completed}/{args.num_requests}")
    print(f"Failed:                {bench_metrics.failed}")
    print(f"Duration:              {duration:.2f}s")
    print(f"Request throughput:    {bench_metrics.request_throughput:.2f} req/s")
    print(f"Output throughput:     {bench_metrics.output_throughput:.2f} tok/s")
    print(f"Total token throughput:{bench_metrics.total_token_throughput:.2f} tok/s")
    print()
    print(f"TTFT (Time to First Token):")
    print(f"  Mean:   {bench_metrics.mean_ttft_ms:.2f} ms")
    print(f"  Median: {bench_metrics.median_ttft_ms:.2f} ms")
    print(f"  P95:    {dict(bench_metrics.percentiles_ttft_ms).get(95.0, 0):.2f} ms")
    print(f"  P99:    {dict(bench_metrics.percentiles_ttft_ms).get(99.0, 0):.2f} ms")
    print()
    print(f"TPOT (Time Per Output Token):")
    print(f"  Mean:   {bench_metrics.mean_tpot_ms:.2f} ms")
    print(f"  Median: {bench_metrics.median_tpot_ms:.2f} ms")
    print(f"  P95:    {dict(bench_metrics.percentiles_tpot_ms).get(95.0, 0):.2f} ms")
    print(f"  P99:    {dict(bench_metrics.percentiles_tpot_ms).get(99.0, 0):.2f} ms")
    print()
    print(f"ITL (Inter-Token Latency):")
    print(f"  Mean:   {bench_metrics.mean_itl_ms:.2f} ms")
    print(f"  Median: {bench_metrics.median_itl_ms:.2f} ms")
    print()
    print(f"E2E Latency:")
    print(f"  Mean:   {bench_metrics.mean_e2el_ms:.2f} ms")
    print(f"  Median: {bench_metrics.median_e2el_ms:.2f} ms")
    print(f"  P95:    {dict(bench_metrics.percentiles_e2el_ms).get(95.0, 0):.2f} ms")
    print(f"  P99:    {dict(bench_metrics.percentiles_e2el_ms).get(99.0, 0):.2f} ms")
    print("=" * 80)

    # Save results in vLLM-compatible format
    from datetime import datetime
    output_data = {
        'date': datetime.now().strftime("%Y%m%d-%H%M%S"),
        'backend': 'vllm_streaming',
        'model_id': args.model,
        'num_prompts': args.num_requests,
        'arrival_rate': args.arrival_rate,
        'input_len': args.input_len,
        'output_len': args.output_len,
        'range_ratio': args.range_ratio,
        'duration': duration,
        'completed': bench_metrics.completed,
        'failed': bench_metrics.failed,
        'request_throughput': bench_metrics.request_throughput,
        'output_throughput': bench_metrics.output_throughput,
        'total_token_throughput': bench_metrics.total_token_throughput,
        'mean_ttft_ms': bench_metrics.mean_ttft_ms,
        'median_ttft_ms': bench_metrics.median_ttft_ms,
        'p95_ttft_ms': dict(bench_metrics.percentiles_ttft_ms).get(95.0, 0),
        'p99_ttft_ms': dict(bench_metrics.percentiles_ttft_ms).get(99.0, 0),
        'mean_tpot_ms': bench_metrics.mean_tpot_ms,
        'median_tpot_ms': bench_metrics.median_tpot_ms,
        'p95_tpot_ms': dict(bench_metrics.percentiles_tpot_ms).get(95.0, 0),
        'p99_tpot_ms': dict(bench_metrics.percentiles_tpot_ms).get(99.0, 0),
        'mean_itl_ms': bench_metrics.mean_itl_ms,
        'median_itl_ms': bench_metrics.median_itl_ms,
        'mean_e2el_ms': bench_metrics.mean_e2el_ms,
        'median_e2el_ms': bench_metrics.median_e2el_ms,
        'p95_e2el_ms': dict(bench_metrics.percentiles_e2el_ms).get(95.0, 0),
        'p99_e2el_ms': dict(bench_metrics.percentiles_e2el_ms).get(99.0, 0),
        'input_lens': [req.prompt_len for req in sample_requests],
        'output_lens': actual_output_lens,
        'ttfts': [output.ttft for output in outputs_sorted],
        'itls': [output.itl for output in outputs_sorted],
        'start_times': [output.start_time for output in outputs_sorted],

        # Batch information for each request
        'batch_info': [dict(step_batch) for step_batch in step_batches],
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Results saved to {args.output}")


if __name__ == "__main__":
    main()
