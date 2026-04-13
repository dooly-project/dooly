import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

try:
    import torch
    from vllm import LLM, SamplingParams
    from vllm.benchmarks.serve import calculate_metrics, BenchmarkMetrics
    from vllm.benchmarks.datasets import SampleRequest
    from vllm.benchmarks.lib.endpoint_request_func import RequestFuncOutput
    VLLM_AVAILABLE = True
except ImportError:
    print("ERROR: vLLM not available. Install with: pip install vllm")
    sys.exit(1)


def convert_to_request_func_output(vllm_output, start_time: float) -> RequestFuncOutput:
    """Convert vLLM RequestOutput to RequestFuncOutput format for calculate_metrics."""
    metrics = vllm_output.metrics

    # Calculate ITLs from metrics
    itls = []
    if metrics.num_generation_tokens > 1:
        # Average ITL across all tokens except first
        avg_itl = (metrics.last_token_ts - metrics.first_token_ts) / metrics.num_generation_tokens
        itls = [avg_itl] * metrics.num_generation_tokens

    # Calculate total latency (E2E from queued to last token)
    # Note: vLLM metrics use monotonic clock, so we use the deltas not absolute values
    # latency = time from request queued until last token generated
    latency = metrics.last_token_ts - metrics.queued_ts

    return RequestFuncOutput(
        generated_text=vllm_output.outputs[0].text if vllm_output.outputs else "",
        success=True,
        latency=latency,
        output_tokens=metrics.num_generation_tokens,
        ttft=metrics.first_token_latency,
        itl=itls,
        tpot=0.0,  # Will be calculated by calculate_metrics
        prompt_len=len(vllm_output.prompt_token_ids),
        error="",
        start_time=start_time,
        input_audio_duration=0.0,
    )


def stream_requests(
    llm: LLM,
    stream_config: Dict,
    warmup: int = 3,
    iterations: int = 10,
    profile_dir: str = './vllm_profiles'
) -> Dict:
    prompt_lens = stream_config['prompt_lens']
    num_prefills = len(prompt_lens)
    output_lens = stream_config['output_lens']

    # Create prompts using explicit token IDs to ensure exact length
    # This guarantees we get exactly the desired token count
    prompt_token_ids_list = []
    for target_len in prompt_lens:
        # Create token IDs from 1 to target_len (avoiding special tokens like 0)
        token_ids = list(range(1, target_len + 1))
        prompt_token_ids_list.append(token_ids)

    print(f"  Created {len(prompt_token_ids_list)} prompts with exact token counts:")
    for i, (target_len, token_ids, out_len) in enumerate(zip(prompt_lens, prompt_token_ids_list, output_lens)):
        print(f"    Prompt {i+1}: {len(token_ids)} tokens (target: {target_len}), output_len: {out_len}")

    # Use max output_len for generation
    max_output_len = max(output_lens)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_output_len,
        ignore_eos=True,
    )
    print(f"  Using max_tokens={max_output_len}")

    # Warmup
    print(f"  Warming up ({warmup} iterations)...")
    for i in range(warmup):
        from vllm.inputs import TokensPrompt
        prompts = [TokensPrompt(prompt_token_ids=token_ids) for token_ids in prompt_token_ids_list]
        llm.generate(prompts, sampling_params=sampling_params)
        torch.cuda.synchronize()

    # Measure with profiling
    print(f"  Profiling prefill + decode phases...")
    latencies = []

    # Start profiling BEFORE the first generate call
    torch.cuda.synchronize()
    llm.start_profile()

    # Run iterations
    per_request_ttfts = []  # Track TTFT for each request
    per_request_itls = []   # Track ITL for each request
    all_request_func_outputs = []  # For calculate_metrics compatibility

    benchmark_start_time = time.perf_counter()
    first_request_arrival_time = None  # Track the first arrival time to make times relative

    for i in range(prompt_lens):
        from vllm.inputs import TokensPrompt
        prompts = [TokensPrompt(prompt_token_ids=token_ids) for token_ids in prompt_token_ids_list]

        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params=sampling_params)
        torch.cuda.synchronize()
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)

        # Extract per-request metrics from vLLM
        print(f"    Per-request metrics:")

        for j, output in enumerate(outputs):
            # Track first arrival time to make all times relative
            if first_request_arrival_time is None:
                first_request_arrival_time = output.metrics.arrival_time

            # Verify actual token count
            actual_prompt_len = len(output.prompt_token_ids)
            print(f"      Request {j+1}: {actual_prompt_len} tokens")
            print(f"         num_generation_tokens: {output.metrics.num_generation_tokens}")
            print(f"         arrival_time: {output.metrics.arrival_time:.3f} s")
            print(f"         queued_ts: {output.metrics.queued_ts:.3f} s", end="")
            print(f"         scheduled_ts: {output.metrics.scheduled_ts:.3f} s")
            print(f"         first_token_ts: {output.metrics.first_token_ts:.3f} s")
            print(f"         last_token_ts: {output.metrics.last_token_ts:.3f} s")
            print(f"         first_token_latency: {output.metrics.first_token_latency * 1000.0} ms")
            print(f"         inter-token latency: {(output.metrics.last_token_ts - output.metrics.first_token_ts) / output.metrics.num_generation_tokens * 1000.0} ms")

            per_request_ttfts.append(output.metrics.first_token_latency)
            per_request_itls.append(((output.metrics.last_token_ts - output.metrics.first_token_ts) / output.metrics.num_generation_tokens))

            # Convert to RequestFuncOutput for calculate_metrics (use relative time)
            relative_start_time = output.metrics.arrival_time - first_request_arrival_time
            request_func_output = convert_to_request_func_output(output, relative_start_time)
            all_request_func_outputs.append(request_func_output)

    benchmark_end_time = time.perf_counter()
    benchmark_duration_s = benchmark_end_time - benchmark_start_time

    llm.stop_profile()

    # Create SampleRequest objects for calculate_metrics
    sample_requests = [
        SampleRequest(
            prompt="",  # We use token IDs directly
            prompt_len=len(token_ids),
            expected_output_len=output_lens[idx] if idx < len(output_lens) else 1,
        )
        for idx, token_ids in enumerate(prompt_token_ids_list)
    ]

    # Use vLLM's calculate_metrics to get standardized metrics
    print(f"  Calculating standardized metrics...")

    # Debug: Print some sample data
    if all_request_func_outputs:
        print(f"  Debug: Total outputs: {len(all_request_func_outputs)}")
        print(f"  Debug: First output - start_time: {all_request_func_outputs[0].start_time:.6f}s, latency: {all_request_func_outputs[0].latency:.6f}s")
        print(f"  Debug: Last output - start_time: {all_request_func_outputs[-1].start_time:.6f}s, latency: {all_request_func_outputs[-1].latency:.6f}s")
        print(f"  Debug: Benchmark duration: {benchmark_duration_s:.6f}s")

    bench_metrics, actual_output_lens = calculate_metrics(
        input_requests=sample_requests * iterations,  # Repeat for each iteration
        outputs=all_request_func_outputs,
        dur_s=benchmark_duration_s,
        tokenizer=None,  # We don't need tokenizer for token counting
        selected_percentiles=[50.0, 95.0, 99.0],
        goodput_config_dict={},
    )

    # Find the generated trace file
    trace_files = sorted(Path(profile_dir).glob("rank*.pt.trace.json"))
    gpu_times = {}
    trace_file_path = None
    if trace_files:
        latest_trace = trace_files[-1]
        print(f"  ✓ Trace saved to: {latest_trace}")
        trace_file_path = str(latest_trace)

        # Parse trace to extract gpu_user_annotation times
        gpu_times = extract_gpu_times_from_trace(trace_file_path)

    result = {
        'latencies': latencies,
        'stats': {
            'mean': np.mean(latencies),
            'std': np.std(latencies),
            'median': np.median(latencies),
            'min': np.min(latencies),
            'max': np.max(latencies),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99),
        },
        'trace_file': trace_file_path,
        'gpu_times': gpu_times,
        'num_prefills': num_prefills,
        'prompt_lens': prompt_lens,
        'output_lens': output_lens,

        'per_request_itl': {
            'mean': np.mean(per_request_itls) * 1000.0,
            'std': np.std(per_request_itls) * 1000.0,
            'median': np.median(per_request_itls) * 1000.0,
            'min': np.min(per_request_itls) * 1000.0,
            'max': np.max(per_request_itls) * 1000.0,
            'p95': np.percentile(per_request_itls, 95) * 1000.0,
            'p99': np.percentile(per_request_itls, 99) * 1000.0,
        },
        'per_request_ttft': {
            'mean': np.mean(per_request_ttfts) * 1000.0,
            'std': np.std(per_request_ttfts) * 1000.0,
            'median': np.median(per_request_ttfts) * 1000.0,
            'min': np.min(per_request_ttfts) * 1000.0,
            'max': np.max(per_request_ttfts) * 1000.0,
            'p95': np.percentile(per_request_ttfts, 95) * 1000.0,
            'p99': np.percentile(per_request_ttfts, 99) * 1000.0,
        },

        # Store vLLM benchmark data for later JSON generation
        # Don't store the dataclass objects directly - extract fields instead
        'vllm_benchmark_data': {
            'metrics': {
                'completed': bench_metrics.completed,
                'failed': bench_metrics.failed,
                'total_input': bench_metrics.total_input,
                'total_output': bench_metrics.total_output,
                'request_throughput': bench_metrics.request_throughput,
                'request_goodput': bench_metrics.request_goodput,
                'output_throughput': bench_metrics.output_throughput,
                'total_token_throughput': bench_metrics.total_token_throughput,
                'mean_ttft_ms': bench_metrics.mean_ttft_ms,
                'median_ttft_ms': bench_metrics.median_ttft_ms,
                'std_ttft_ms': bench_metrics.std_ttft_ms,
                'percentiles_ttft_ms': bench_metrics.percentiles_ttft_ms,
                'mean_tpot_ms': bench_metrics.mean_tpot_ms,
                'median_tpot_ms': bench_metrics.median_tpot_ms,
                'std_tpot_ms': bench_metrics.std_tpot_ms,
                'percentiles_tpot_ms': bench_metrics.percentiles_tpot_ms,
                'mean_itl_ms': bench_metrics.mean_itl_ms,
                'median_itl_ms': bench_metrics.median_itl_ms,
                'std_itl_ms': bench_metrics.std_itl_ms,
                'percentiles_itl_ms': bench_metrics.percentiles_itl_ms,
                'mean_e2el_ms': bench_metrics.mean_e2el_ms,
                'median_e2el_ms': bench_metrics.median_e2el_ms,
                'std_e2el_ms': bench_metrics.std_e2el_ms,
                'percentiles_e2el_ms': bench_metrics.percentiles_e2el_ms,
                'max_output_tokens_per_s': bench_metrics.max_output_tokens_per_s,
                'max_concurrent_requests': bench_metrics.max_concurrent_requests,
                'rtfx': bench_metrics.rtfx,
            },
            'outputs': {
                'ttfts': [output.ttft for output in all_request_func_outputs],
                'itls': [output.itl for output in all_request_func_outputs],
                'start_times': [output.start_time for output in all_request_func_outputs],
                'generated_texts': [output.generated_text for output in all_request_func_outputs],
                'errors': [output.error for output in all_request_func_outputs],
                'prompt_lens': [output.prompt_len for output in all_request_func_outputs],
                'output_tokens': [output.output_tokens for output in all_request_func_outputs],
            }
        },
    }
    return result



def measure_batch_request(
    llm: LLM,
    batch_config: Dict,
    warmup: int = 3,
    iterations: int = 10,
    profile_dir: str = './vllm_profiles'
) -> Dict:
    prompt_lens = batch_config['prompt_lens']
    num_prefills = len(prompt_lens)
    output_lens = batch_config['output_lens']

    # Create prompts using explicit token IDs to ensure exact length
    # This guarantees we get exactly the desired token count
    prompt_token_ids_list = []
    for target_len in prompt_lens:
        # Create token IDs from 1 to target_len (avoiding special tokens like 0)
        token_ids = list(range(1, target_len + 1))
        prompt_token_ids_list.append(token_ids)

    print(f"  Created {len(prompt_token_ids_list)} prompts with exact token counts:")
    for i, (target_len, token_ids, out_len) in enumerate(zip(prompt_lens, prompt_token_ids_list, output_lens)):
        print(f"    Prompt {i+1}: {len(token_ids)} tokens (target: {target_len}), output_len: {out_len}")

    # Use max output_len for generation
    max_output_len = max(output_lens)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_output_len,
        ignore_eos=True,
    )
    print(f"  Using max_tokens={max_output_len}")

    # Warmup - run WITHOUT profiling
    print(f"  Warming up ({warmup} iterations)...")
    for i in range(warmup):
        from vllm.inputs import TokensPrompt
        prompts = [TokensPrompt(prompt_token_ids=token_ids) for token_ids in prompt_token_ids_list]
        llm.generate(prompts, sampling_params=sampling_params)
        torch.cuda.synchronize()

    # Measure with profiling
    print(f"  Profiling prefill + decode phases...")
    latencies = []

    # Start profiling BEFORE the first generate call
    torch.cuda.synchronize()
    llm.start_profile()

    # Run iterations
    per_request_ttfts = []  # Track TTFT for each request
    per_request_itls = []   # Track ITL for each request
    all_request_func_outputs = []  # For calculate_metrics compatibility

    benchmark_start_time = time.perf_counter()
    first_request_arrival_time = None  # Track the first arrival time to make times relative

    for i in range(iterations):
        # Pass token IDs as TextTokensPrompt objects
        from vllm.inputs import TokensPrompt
        prompts = [TokensPrompt(prompt_token_ids=token_ids) for token_ids in prompt_token_ids_list]
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params=sampling_params)
        torch.cuda.synchronize()
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)

        # Extract per-request metrics from vLLM
        print(f"    Per-request metrics:")

        for j, output in enumerate(outputs):
            # Track first arrival time to make all times relative
            if first_request_arrival_time is None:
                first_request_arrival_time = output.metrics.arrival_time

            # Verify actual token count
            actual_prompt_len = len(output.prompt_token_ids)
            print(f"      Request {j+1}: {actual_prompt_len} tokens")
            print(f"         num_generation_tokens: {output.metrics.num_generation_tokens}")
            print(f"         arrival_time: {output.metrics.arrival_time:.3f} s")
            print(f"         queued_ts: {output.metrics.queued_ts:.3f} s")
            print(f"         scheduled_ts: {output.metrics.scheduled_ts:.3f} s")
            print(f"         first_token_ts: {output.metrics.first_token_ts:.3f} s")
            print(f"         last_token_ts: {output.metrics.last_token_ts:.3f} s")
            print(f"         first_token_latency: {output.metrics.first_token_latency * 1000.0} ms")
            print(f"         inter-token latency: {(output.metrics.last_token_ts - output.metrics.first_token_ts) / output.metrics.num_generation_tokens * 1000.0} ms")

            per_request_ttfts.append(output.metrics.first_token_latency)
            per_request_itls.append(((output.metrics.last_token_ts - output.metrics.first_token_ts) / output.metrics.num_generation_tokens))

            # Convert to RequestFuncOutput for calculate_metrics (use relative time)
            relative_start_time = output.metrics.arrival_time - first_request_arrival_time
            request_func_output = convert_to_request_func_output(output, relative_start_time)
            all_request_func_outputs.append(request_func_output)

    benchmark_end_time = time.perf_counter()
    benchmark_duration_s = benchmark_end_time - benchmark_start_time

    llm.stop_profile()

    # Create SampleRequest objects for calculate_metrics
    sample_requests = [
        SampleRequest(
            prompt="",  # We use token IDs directly
            prompt_len=len(token_ids),
            expected_output_len=output_lens[idx] if idx < len(output_lens) else 1,
        )
        for idx, token_ids in enumerate(prompt_token_ids_list)
    ]

    # Use vLLM's calculate_metrics to get standardized metrics
    print(f"  Calculating standardized metrics...")

    # Debug: Print some sample data
    if all_request_func_outputs:
        print(f"  Debug: Total outputs: {len(all_request_func_outputs)}")
        print(f"  Debug: First output - start_time: {all_request_func_outputs[0].start_time:.6f}s, latency: {all_request_func_outputs[0].latency:.6f}s")
        print(f"  Debug: Last output - start_time: {all_request_func_outputs[-1].start_time:.6f}s, latency: {all_request_func_outputs[-1].latency:.6f}s")
        print(f"  Debug: Benchmark duration: {benchmark_duration_s:.6f}s")

    bench_metrics, actual_output_lens = calculate_metrics(
        input_requests=sample_requests * iterations,  # Repeat for each iteration
        outputs=all_request_func_outputs,
        dur_s=benchmark_duration_s,
        tokenizer=None,  # We don't need tokenizer for token counting
        selected_percentiles=[50.0, 95.0, 99.0],
        goodput_config_dict={},
    )

    # Find the generated trace file
    trace_files = sorted(Path(profile_dir).glob("rank*.pt.trace.json"))
    gpu_times = {}
    trace_file_path = None
    if trace_files:
        latest_trace = trace_files[-1]
        print(f"  ✓ Trace saved to: {latest_trace}")
        trace_file_path = str(latest_trace)

        # Parse trace to extract gpu_user_annotation times
        gpu_times = extract_gpu_times_from_trace(trace_file_path)

    result = {
        'latencies': latencies,
        'stats': {
            'mean': np.mean(latencies),
            'std': np.std(latencies),
            'median': np.median(latencies),
            'min': np.min(latencies),
            'max': np.max(latencies),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99),
        },
        'trace_file': trace_file_path,
        'gpu_times': gpu_times,
        'num_prefills': num_prefills,
        'prompt_lens': prompt_lens,
        'output_lens': output_lens,

        'per_request_itl': {
            'mean': np.mean(per_request_itls) * 1000.0,
            'std': np.std(per_request_itls) * 1000.0,
            'median': np.median(per_request_itls) * 1000.0,
            'min': np.min(per_request_itls) * 1000.0,
            'max': np.max(per_request_itls) * 1000.0,
            'p95': np.percentile(per_request_itls, 95) * 1000.0,
            'p99': np.percentile(per_request_itls, 99) * 1000.0,
        },
        'per_request_ttft': {
            'mean': np.mean(per_request_ttfts) * 1000.0,
            'std': np.std(per_request_ttfts) * 1000.0,
            'median': np.median(per_request_ttfts) * 1000.0,
            'min': np.min(per_request_ttfts) * 1000.0,
            'max': np.max(per_request_ttfts) * 1000.0,
            'p95': np.percentile(per_request_ttfts, 95) * 1000.0,
            'p99': np.percentile(per_request_ttfts, 99) * 1000.0,
        },

        # Store vLLM benchmark data for later JSON generation
        # Don't store the dataclass objects directly - extract fields instead
        'vllm_benchmark_data': {
            'metrics': {
                'completed': bench_metrics.completed,
                'failed': bench_metrics.failed,
                'total_input': bench_metrics.total_input,
                'total_output': bench_metrics.total_output,
                'request_throughput': bench_metrics.request_throughput,
                'request_goodput': bench_metrics.request_goodput,
                'output_throughput': bench_metrics.output_throughput,
                'total_token_throughput': bench_metrics.total_token_throughput,
                'mean_ttft_ms': bench_metrics.mean_ttft_ms,
                'median_ttft_ms': bench_metrics.median_ttft_ms,
                'std_ttft_ms': bench_metrics.std_ttft_ms,
                'percentiles_ttft_ms': bench_metrics.percentiles_ttft_ms,
                'mean_tpot_ms': bench_metrics.mean_tpot_ms,
                'median_tpot_ms': bench_metrics.median_tpot_ms,
                'std_tpot_ms': bench_metrics.std_tpot_ms,
                'percentiles_tpot_ms': bench_metrics.percentiles_tpot_ms,
                'mean_itl_ms': bench_metrics.mean_itl_ms,
                'median_itl_ms': bench_metrics.median_itl_ms,
                'std_itl_ms': bench_metrics.std_itl_ms,
                'percentiles_itl_ms': bench_metrics.percentiles_itl_ms,
                'mean_e2el_ms': bench_metrics.mean_e2el_ms,
                'median_e2el_ms': bench_metrics.median_e2el_ms,
                'std_e2el_ms': bench_metrics.std_e2el_ms,
                'percentiles_e2el_ms': bench_metrics.percentiles_e2el_ms,
                'max_output_tokens_per_s': bench_metrics.max_output_tokens_per_s,
                'max_concurrent_requests': bench_metrics.max_concurrent_requests,
                'rtfx': bench_metrics.rtfx,
            },
            'outputs': {
                'ttfts': [output.ttft for output in all_request_func_outputs],
                'itls': [output.itl for output in all_request_func_outputs],
                'start_times': [output.start_time for output in all_request_func_outputs],
                'generated_texts': [output.generated_text for output in all_request_func_outputs],
                'errors': [output.error for output in all_request_func_outputs],
                'prompt_lens': [output.prompt_len for output in all_request_func_outputs],
                'output_tokens': [output.output_tokens for output in all_request_func_outputs],
            }
        },
    }
    return result


def extract_gpu_times_from_trace(trace_path: str) -> Dict:

    try:
        with open(trace_path, 'r') as f:
            trace = json.load(f)

        gpu_times = {
            'prefill': [],
            'decode': [],
        }

        import re
        for event in trace.get('traceEvents', []):
            if event.get('cat') == 'gpu_user_annotation':
                name = event.get('name', '')
                if 'execute_context' in name:
                    dur_ms = event.get('dur', 0) / 1000

                    # Full pattern: execute_context_<prefill_batch>(<prefill_tokens>)_generation_<decode_batch>(<decode_tokens>)
                    match = re.search(r'execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)', name)
                    if match:
                        prefill_batch = int(match.group(1))
                        prefill_tokens = int(match.group(2))
                        decode_batch = int(match.group(3))
                        decode_tokens = int(match.group(4))

                        # Categorize based on what work is being done
                        if prefill_tokens > 0:
                            # This is prefill work (may also have some decode)
                            gpu_times['prefill'].append(dur_ms)
                        elif decode_tokens > 0:
                            # Pure decode work
                            gpu_times['decode'].append(dur_ms)

        # Calculate stats
        result = {}
        if gpu_times['prefill']:
            result['prefill'] = {
                'mean': np.mean(gpu_times['prefill']),
                'median': np.median(gpu_times['prefill']),
                'min': np.min(gpu_times['prefill']),
                'max': np.max(gpu_times['prefill']),
                'count': len(gpu_times['prefill']),
                'values': gpu_times['prefill'],
            }
        if gpu_times['decode']:
            result['decode'] = {
                'mean': np.mean(gpu_times['decode']),
                'median': np.median(gpu_times['decode']),
                'min': np.min(gpu_times['decode']),
                'max': np.max(gpu_times['decode']),
                'count': len(gpu_times['decode']),
                'values': gpu_times['decode'],
            }

        return result

    except Exception as e:
        print(f"  ⚠ Error parsing trace: {e}")
        return {}


def create_test_cases() -> List[Tuple[str, Dict]]:
    """Create test configurations matching DoolySim tests."""

    return [
        ("Single Prefill 32", {
            'prompt_lens': [32],
            'output_lens': [1],
            'description': 'Single 8-token prefill + 1 decode'
        }),

        ("Batch of 4 Prefill 32", {
            'prompt_lens': [8, 8, 8, 8],
            'output_lens': [1, 1, 1, 1],
            'description': 'Multiple 8-token prefill + 1 decode'
        }),

        # ("Single Prefill 3072", {
        #     'prompt_lens': [3072],
        #     'output_lens': [1],
        #     'description': 'Single 3072-token prefill + 1 decode'
        # }),

        # ("Single Prefill 512", {
        #     'prompt_lens': [512],
        #     'output_lens': [1],
        #     'description': 'Single 512-token prefill + 1 decode'
        # }),

        # ("Batch Prefill", {
        #     'prompt_lens': [128, 256, 512],
        #     'output_lens': [1, 1, 1],
        #     'description': '3 prefill requests'
        # }),

        # ("Single Prefill Short", {
        #     'prompt_lens': [32],
        #     'output_lens': [1],
        #     'description': 'Single request with 32 output tokens'
        # }),

        # ("Batch Decode", {
        #     'prompt_lens': [256, 512, 768, 1024],
        #     'output_lens': [32, 32, 32, 32],
        #     'description': '4 decode requests (32 token each)'
        # }),
    ]


def main():
    """Main measurement entry point."""

    import argparse
    parser = argparse.ArgumentParser(description="Measure vLLM latencies directly")
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B',
                       help='Model to benchmark')
    parser.add_argument('--warmup', type=int, default=3,
                       help='Number of warmup iterations')
    parser.add_argument('--iterations', type=int, default=10,
                       help='Number of measurement iterations')
    parser.add_argument('--output', type=str, default='vllm_measurements.json',
                       help='Output JSON file')
    parser.add_argument('--gpu-memory', type=float, default=0.9,
                       help='GPU memory utilization')
    args = parser.parse_args()

    print("=" * 80)
    print("DIRECT VLLM LATENCY MEASUREMENT")
    print("=" * 80)
    print(f"Model:       {args.model}")
    print(f"Warmup:      {args.warmup}")
    print(f"Iterations:  {args.iterations}")
    print(f"Output:      {args.output}")
    print("=" * 80)
    print()

    # Initialize vLLM
    print("Initializing vLLM (this may take a minute)...")
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=0.9,
        max_num_seqs=32,
        max_model_len=8192,
        max_num_batched_tokens=8192,
        load_format="dummy",
        enable_prefix_caching=False,
        disable_log_stats=False,
        enforce_eager=False,  # Disable CUDA graphs for accurate timing
        attention_backend="FLASHINFER",
        profiler_config={
            'profiler': 'torch',
            'torch_profiler_dir': './vllm_profiles',
            'torch_profiler_use_gzip': False,
            'torch_profiler_with_stack': True,
        },
    )
    print("✓ vLLM initialized\n")

    # Run tests
    test_cases = create_test_cases()
    all_results = []

    for test_name, config in test_cases:
        print("=" * 80)
        print(f"TEST: {test_name}")
        print("=" * 80)
        print(f"Config: {config['description']}")
        print(f"Prompts: {config['prompt_lens']}")
        print(f"Outputs: {config['output_lens']}")
        print()

        print(f"Measuring... (warmup={args.warmup}, iterations={args.iterations})")
        result = measure_batch_request(
            llm=llm,
            batch_config=config,
            warmup=args.warmup,
            iterations=args.iterations,
            profile_dir='./vllm_profiles'
        )

        stats = result['per_request_ttft']
        print(f"\n TTFT (vLLM Official Timed Metrics):")
        print(f"  Mean:   {stats['mean']:.3f} ± {stats['std']:.3f} ms")
        print(f"  Median: {stats['median']:.3f} ms")
        print(f"  Min:    {stats['min']:.3f} ms")
        print(f"  Max:    {stats['max']:.3f} ms")
        print(f"  P95:    {stats['p95']:.3f} ms")
        print(f"  P99:    {stats['p99']:.3f} ms")
        
        stats = result['per_request_itl']
        print(f"\n ITL (Inter-Token Latency):")
        print(f"  Mean:   {stats['mean']:.3f} ± {stats['std']:.3f} ms")
        print(f"  Median: {stats['median']:.3f} ms")
        print(f"  Min:    {stats['min']:.3f} ms")
        print(f"  Max:    {stats['max']:.3f} ms")
        print(f"  P95:    {stats['p95']:.3f} ms")
        print(f"  P99:    {stats['p99']:.3f} ms")
        
        # Display GPU times from trace
        if result.get('gpu_times'):
            num_prefills = result.get('num_prefills')
            gpu_times = result['gpu_times']
            print(f"\nGPU execution time (from trace gpu_user_annotation):")
            if 'prefill' in gpu_times:
                prefill = gpu_times['prefill']
                print(f"  Number of GPU forward passes detected: {prefill['count']}")
                print(f"  Individual GPU times: {[f'{t:.3f}ms' for t in prefill['values']]}")
                print(f"  Prefill mean:   {prefill['mean']:.3f} ms")
                print(f"  Prefill median: {prefill['median']:.3f} ms")
                print(f"  Prefill range:  {prefill['min']:.3f} - {prefill['max']:.3f} ms")
                if prefill['count'] > 1:
                    print(f"  ⚠ Note: Multiple batches detected - requests were NOT all batched together!")
            if 'decode' in gpu_times:
                decode = gpu_times['decode']
                print(f"  Decode mean:    {decode['mean']:.3f} ms ({decode['count']} samples)")

            # Calculate overhead
            if 'prefill' in gpu_times:
                gpu_mean = gpu_times['prefill']['mean'] * result['num_prefills']
                wall_clock_mean = stats['mean']
                overhead_ms = wall_clock_mean - gpu_mean
                overhead_pct = (overhead_ms / gpu_mean) * 100
                print(f"\nOverhead analysis:")
                print(f"  GPU time:      {gpu_mean:.3f} ms (what your profiler captures)")
                print(f"  Wall-clock:    {wall_clock_mean:.3f} ms (what client measures)")
                print(f"  Overhead:      {overhead_ms:.3f} ms ({overhead_pct:+.1f}%)")
                print(f"  Calibration:   multiply GPU predictions by {wall_clock_mean/gpu_mean:.3f}x")

        print()

        all_results.append({
            'test_name': test_name,
            **result
        })

    # Save results
    print("=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    output_data = {
        'model': args.model,
        'config': {
            'warmup': args.warmup,
            'iterations': args.iterations,
            'gpu_memory_utilization': args.gpu_memory,
        },
        'results': all_results,
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Results saved to {args.output}")

    # Also save in vllm_raw.json compatible format for each test
    from datetime import datetime
    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")

    for result in all_results:
        test_name = result['test_name'].replace(' ', '_').lower()
        compatible_filename = args.output.replace('.json', f'_{test_name}_vllm_compatible.json')

        if 'vllm_benchmark_data' not in result:
            print(f"  ⚠ Skipping {test_name}: no vLLM benchmark data available")
            continue

        bench_data = result['vllm_benchmark_data']
        metrics = bench_data['metrics']
        outputs = bench_data['outputs']

        # Create vllm_raw.json compatible format
        vllm_compatible_data = {
            'date': date_str,
            'endpoint_type': 'vllm',
            'backend': 'vllm',
            'label': result['test_name'],
            'model_id': args.model,
            'tokenizer_id': args.model,
            'num_prompts': result.get('num_prefills', 1),
            'request_rate': 'inf',
            'burstiness': 1.0,
            'max_concurrency': None,
            'duration': result.get('stats', {}).get('mean', 0) / 1000.0,
            'completed': metrics['completed'],
            'failed': metrics['failed'],
            'total_input_tokens': metrics['total_input'],
            'total_output_tokens': metrics['total_output'],
            'request_throughput': metrics['request_throughput'],
            'request_goodput': metrics['request_goodput'],
            'output_throughput': metrics['output_throughput'],
            'total_token_throughput': metrics['total_token_throughput'],
            'input_lens': result.get('prompt_lens', []),
            'output_lens': outputs['output_tokens'],
            'ttfts': outputs['ttfts'],
            'itls': outputs['itls'],
            'start_times': outputs['start_times'],
            'generated_texts': outputs['generated_texts'],
            'errors': outputs['errors'],
            'max_output_tokens_per_s': metrics['max_output_tokens_per_s'],
            'max_concurrent_requests': metrics['max_concurrent_requests'],
            'rtfx': metrics['rtfx'],
            'mean_ttft_ms': metrics['mean_ttft_ms'],
            'median_ttft_ms': metrics['median_ttft_ms'],
            'std_ttft_ms': metrics['std_ttft_ms'],
            'p99_ttft_ms': dict(metrics['percentiles_ttft_ms']).get(99.0, 0.0),
            'mean_tpot_ms': metrics['mean_tpot_ms'],
            'median_tpot_ms': metrics['median_tpot_ms'],
            'std_tpot_ms': metrics['std_tpot_ms'],
            'p99_tpot_ms': dict(metrics['percentiles_tpot_ms']).get(99.0, 0.0),
            'mean_itl_ms': metrics['mean_itl_ms'],
            'median_itl_ms': metrics['median_itl_ms'],
            'std_itl_ms': metrics['std_itl_ms'],
            'p99_itl_ms': dict(metrics['percentiles_itl_ms']).get(99.0, 0.0),
        }

        with open(compatible_filename, 'w') as f:
            json.dump(vllm_compatible_data, f, indent=2)

        print(f"✓ vLLM-compatible format saved to {compatible_filename}")

    print()

    # Print summary table
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Test Name':<25} {'Mean (ms)':>12} {'Std (ms)':>12} {'Median (ms)':>12} {'GPU Prefill Mean':>20} {'GPU Prefill Median':>20}")
    print("-" * 80)

    print("TTFT (Time to First Token):")

    
    
    for result in all_results:
        stats = result['per_request_ttft']
        num_prefills = result.get('num_prefills')
        gpu_times = result['gpu_times']
   
        print(f"{result['test_name']:<25} "
              f"{stats['mean'] :>12.3f} "
            #   f"{stats['mean'] - 10:>12.3f} "
              f"{stats['std']:>12.3f} "
              f"{stats['median']:>12.3f} "
            #   f"{stats['median'] - 10:>12.3f} "
              f"{gpu_times['prefill']['mean']:>12.3f} ms (GPU) "
              f"{gpu_times['prefill']['median']:>12.3f} ms (GPU)")

    print("ITL (Inter-Token Latency):")
    for result in all_results:
        gpu_times = result['gpu_times']
        stats = result['per_request_itl']
        gpu_decode_str = ""
        if 'decode' in gpu_times:
            gpu_decode_str = f"{gpu_times['decode']['mean']:>12.3f} ms (GPU) {gpu_times['decode']['median']:>12.3f} ms (GPU)"
        print(f"{result['test_name']:<25} "
              f"{stats['mean']:>12.3f} "
              f"{stats['std']:>12.3f} "
              f"{stats['median']:>12.3f} "
              f"{gpu_decode_str}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
