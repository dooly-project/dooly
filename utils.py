from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class SimulationStep:
    """Record of a single simulation step."""
    step_id: int
    sim_time: float

    # Scheduling info
    num_scheduled_new: int
    num_scheduled_cached: int
    total_tokens_scheduled: int
    scheduled_req_ids: List[str]

    # Block usage
    free_blocks: int

    # Finished/preempted
    finished_req_ids: List[str]
    preempted_req_ids: List[str]

    # Request states
    waiting_count: int
    running_count: int


@dataclass
class SimulationResult:
    """Results from a simulation run."""
    steps: List[SimulationStep] = field(default_factory=list)
    total_requests: int = 0
    total_steps: int = 0
    total_tokens_generated: int = 0
    preemption_count: int = 0

    # Per-request stats
    request_latencies: Dict[str, float] = field(default_factory=dict)  # req_id -> total latency
    request_ttft: Dict[str, float] = field(default_factory=dict)  # req_id -> time to first token

