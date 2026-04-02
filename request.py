from attr import dataclass
from typing import Iterator, List
import random

from vllm.v1.request import Request
from vllm.sampling_params import SamplingParams
from vllm.v1.outputs import ModelRunnerOutput

@dataclass
class RequestConfig:
    """Configuration for a single request."""
    request_id: str
    prompt_token_ids: List[int]
    max_tokens: int
    arrival_time: float  
    
class RequestStream:
    def __init__(
        self,
        num_requests: int,
        arrival_rate: float,  # requests per second
        prompt_len_range: tuple[int, int] = (10, 100),
        output_len_range: tuple[int, int] = (10, 50),
        seed: int = 42,
    ):
        self.num_requests = num_requests
        self.arrival_rate = arrival_rate
        self.prompt_len_range = prompt_len_range
        self.output_len_range = output_len_range
        self.rng = random.Random(seed)

    def generate(self) -> Iterator[RequestConfig]:
        """Generate requests with Poisson arrival times."""
        current_time = 0.0

        for i in range(self.num_requests):
            if self.arrival_rate > 0:
                inter_arrival = self.rng.expovariate(self.arrival_rate)
            else:
                inter_arrival = 0.0
            current_time += inter_arrival

            prompt_len = self.rng.randint(*self.prompt_len_range)
            output_len = self.rng.randint(*self.output_len_range)

            yield RequestConfig(
                request_id=f"req_{i}",
                prompt_token_ids=list(range(100, 100 + prompt_len)),
                max_tokens=output_len,
                arrival_time=current_time,
            )

def make_request(self, config: RequestConfig):
    """Create a request object."""""
    request_block_hasher = self.llm.llm_engine.engine_core.engine_core.request_block_hasher
    req = Request(
        request_id=config.request_id,
        prompt_token_ids=config.prompt_token_ids,
        sampling_params=SamplingParams(max_tokens=config.max_tokens),
        pooling_params=None,
        block_hasher=request_block_hasher,
    )
    
    return req


def create_mock_output(self, scheduler_output, token_id: int = 1000):
    """Create mock ModelRunnerOutput simulating one token generated per request."""
    req_ids = list(scheduler_output.num_scheduled_tokens.keys())

    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=[[token_id] for _ in req_ids],
    )