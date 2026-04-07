"""Request generator using vLLM's dataset generation for reproducibility.

This ensures DoolySim and vLLM benchmarks use identical requests.
"""

from typing import List, Tuple
from dataclasses import dataclass
import numpy as np

try:
    from vllm.benchmarks.datasets import RandomDataset, ShareGPTDataset, SampleRequest
    from vllm import LLM
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    SampleRequest = None


@dataclass
class Request:
    """DoolySim request format."""
    request_id: str
    prompt_token_ids: List[int]
    max_tokens: int
    arrival_time: float

def generate_vllm_requests_from_instance( 
    llm: LLM,
    num_requests: int,
    arrival_rate: float,
    input_len: int,
    output_len: int,
    range_ratio: float = 0.0,
    seed: int = 42,
) -> Tuple[List[Request], List[SampleRequest]]:

    tokenizer = llm.get_tokenizer()

    # Create vLLM's RandomDataset
    dataset = RandomDataset(random_seed=seed, disable_shuffle=True)

    print(f"Generating {num_requests} requests with vLLM RandomDataset...")
    print(f"  Input length: {input_len} (range_ratio={range_ratio})")
    print(f"  Output length: {output_len} (range_ratio={range_ratio})")
    print(f"  Seed: {seed}")

    # Generate requests using vLLM's method
    vllm_requests = dataset.sample(
        tokenizer=tokenizer,
        num_requests=num_requests,
        input_len=input_len,
        output_len=output_len,
        burstiness=5.0,
        range_ratio=range_ratio,
        request_id_prefix="req_",
    )

    print(f"Generated {len(vllm_requests)} requests")

    # Generate arrival times using Poisson process
    rng = np.random.default_rng(seed)
    inter_arrival_times = rng.exponential(scale=1.0 / arrival_rate, size=num_requests)
    arrival_times = np.cumsum(inter_arrival_times)

    # Convert to DoolySim format
    dooly_requests = []
    for vllm_req, arrival_time in zip(vllm_requests, arrival_times):
        # Tokenize the prompt to get token IDs
        token_ids = tokenizer.encode(vllm_req.prompt)

        dooly_req = Request(
            request_id=vllm_req.request_id,
            prompt_token_ids=token_ids,
            max_tokens=vllm_req.expected_output_len,
            arrival_time=arrival_time,
        )
        dooly_requests.append(dooly_req)

    # Verify lengths match
    print("\nRequest verification:")
    print(f"  vLLM requests: {len(vllm_requests)}")
    print(f"  DoolySim requests: {len(dooly_requests)}")
    print(f"  Sample prompt lengths: {[r.prompt_len for r in vllm_requests[:5]]}")
    print(f"  Sample output lengths: {[r.expected_output_len for r in vllm_requests[:5]]}")
    print(f"  Sample arrival times: {[f'{t:.3f}' for t in arrival_times[:5]]}")

    return dooly_requests, vllm_requests


def generate_sharegpt_requests_from_instance(
    llm: LLM,
    num_requests: int,
    arrival_rate: float,
    sharegpt_path: str,
    custom_output_len: int | None = None,
    seed: int = 42,
) -> Tuple[List[Request], List[SampleRequest]]:
    """Generate requests from ShareGPT dataset using vLLM's ShareGPTDataset.

    Args:
        llm: vLLM LLM instance (for tokenizer)
        num_requests: Number of requests to sample
        custom_output_len: Custom output length for all requests
        arrival_rate: Requests per second (for Poisson arrivals)
        sharegpt_path: Path to ShareGPT JSON file
        seed: Random seed

    Returns:
        (dooly_requests, vllm_requests) tuple
    """
    tokenizer = llm.get_tokenizer()

    # Create vLLM's ShareGPTDataset
    dataset = ShareGPTDataset(dataset_path=sharegpt_path, random_seed=seed)

    print(f"Sampling {num_requests} requests from ShareGPT dataset...")
    print(f"  Dataset path: {sharegpt_path}")
    print(f"  Seed: {seed}")

    # Sample requests from ShareGPT
    vllm_requests = dataset.sample(
        tokenizer=tokenizer,
        num_requests=num_requests,
        output_len=custom_output_len,
        request_id_prefix="req_",
    )

    print(f"Sampled {len(vllm_requests)} requests from ShareGPT")

    # Generate arrival times using Poisson process
    rng = np.random.default_rng(seed)
    inter_arrival_times = rng.exponential(scale=1.0 / arrival_rate, size=len(vllm_requests))
    arrival_times = np.cumsum(inter_arrival_times)

    # Convert to DoolySim format
    dooly_requests = []
    for i, vllm_req in enumerate(vllm_requests):
        # Tokenize the prompt
        token_ids = tokenizer.encode(vllm_req.prompt)

        dooly_req = Request(
            request_id=vllm_req.request_id,
            prompt_token_ids=token_ids,
            max_tokens=vllm_req.expected_output_len,
            arrival_time=arrival_times[i],
        )
        dooly_requests.append(dooly_req)

    print(f"\n✓ Generated {len(dooly_requests)} requests")
    print(f"  Sample prompt lengths: {[len(r.prompt_token_ids) for r in dooly_requests[:5]]}")
    print(f"  Sample output lengths: {[r.max_tokens for r in dooly_requests[:5]]}")
    print(f"  Sample arrival times: {[f'{t:.3f}' for t in arrival_times[:5]]}")

    return dooly_requests, vllm_requests


def generate_vllm_requests(
    model: str,
    num_requests: int,
    arrival_rate: float,
    input_len: int,
    output_len: int,
    range_ratio: float = 0.0,
    seed: int = 42,
) -> Tuple[List[Request], List[SampleRequest]]:
    if not VLLM_AVAILABLE:
        raise ImportError("vLLM is not installed. Install with: pip install vllm")

    print(f"Initializing tokenizer for {model}...")
    # Initialize LLM to get tokenizer
    llm = LLM(
        model=model,
        download_dir=None,
        load_format="dummy",  # Don't load weights, just tokenizer
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()

    # Create vLLM's RandomDataset
    dataset = RandomDataset(random_seed=seed, disable_shuffle=True)

    print(f"Generating {num_requests} requests with vLLM RandomDataset...")
    print(f"  Input length: {input_len} (range_ratio={range_ratio})")
    print(f"  Output length: {output_len} (range_ratio={range_ratio})")
    print(f"  Seed: {seed}")

    # Generate requests using vLLM's method
    vllm_requests = dataset.sample(
        tokenizer=tokenizer,
        num_requests=num_requests,
        input_len=input_len,
        output_len=output_len,
        burstiness=5.0,
        range_ratio=range_ratio,
        request_id_prefix="req_",
    )

    print(f"Generated {len(vllm_requests)} requests")

    # Generate arrival times using Poisson process
    rng = np.random.default_rng(seed)
    inter_arrival_times = rng.exponential(scale=1.0 / arrival_rate, size=num_requests)
    arrival_times = np.cumsum(inter_arrival_times)

    # Convert to DoolySim format
    dooly_requests = []
    for vllm_req, arrival_time in zip(vllm_requests, arrival_times):
        # Tokenize the prompt to get token IDs
        token_ids = tokenizer.encode(vllm_req.prompt)

        dooly_req = Request(
            request_id=vllm_req.request_id,
            prompt_token_ids=token_ids,
            max_tokens=vllm_req.expected_output_len,
            arrival_time=arrival_time,
        )
        dooly_requests.append(dooly_req)

    # Verify lengths match
    print("\nRequest verification:")
    print(f"  vLLM requests: {len(vllm_requests)}")
    print(f"  DoolySim requests: {len(dooly_requests)}")
    print(f"  Sample prompt lengths: {[r.prompt_len for r in vllm_requests[:5]]}")
    print(f"  Sample output lengths: {[r.expected_output_len for r in vllm_requests[:5]]}")
    print(f"  Sample arrival times: {[f'{t:.3f}' for t in arrival_times[:5]]}")

    return dooly_requests, vllm_requests


def export_vllm_dataset_jsonl(
    vllm_requests: List[SampleRequest],
    arrival_times: List[float],
    output_path: str,
):
    """Export vLLM requests to JSONL format for vllm bench serve.

    Args:
        vllm_requests: List of SampleRequest from generate_vllm_requests()
        arrival_times: List of arrival times for each request
        output_path: Path to output JSONL file
    """
    import json
    import os

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, 'w') as f:
        for req, arrival_time in zip(vllm_requests, arrival_times):
            entry = {
                "prompt": req.prompt,
                "output_len": req.expected_output_len,
                "request_id": req.request_id,
                "prompt_len": req.prompt_len,
                "arrival_time": arrival_time,
            }
            f.write(json.dumps(entry) + "\n")

    print(f"Exported {len(vllm_requests)} requests to {output_path}")


if __name__ == "__main__":
    # Test the generator
    dooly_reqs, vllm_reqs = generate_vllm_requests(
        model="meta-llama/Llama-3.1-8B",
        num_requests=10,
        arrival_rate=5.0,
        input_len=1024,
        output_len=128,
        range_ratio=0.5,
        seed=42,
    )

    print("\nSample DoolySim request:")
    print(f"  ID: {dooly_reqs[0].request_id}")
    print(f"  Prompt tokens: {len(dooly_reqs[0].prompt_token_ids)}")
    print(f"  Max tokens: {dooly_reqs[0].max_tokens}")
    print(f"  Arrival time: {dooly_reqs[0].arrival_time:.3f}s")

    print("\nSample vLLM request:")
    print(f"  ID: {vllm_reqs[0].request_id}")
    print(f"  Prompt length: {vllm_reqs[0].prompt_len}")
    print(f"  Expected output: {vllm_reqs[0].expected_output_len}")
