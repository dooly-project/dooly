from .utils import SimulationResult, SimulationStep
from .request import RequestConfig, RequestStream, make_request, create_mock_output
from .emulator import emulate_init 

from typing import Dict, Iterator

class DoolySim:
    def __init__(
        self,
        model: str = "gpt2",
        args: Dict = {"max_model_len": 512, "gpu_memory_utilization": 0.0003439735828288387, "max_num_seqs": 10, "max_num_batched_tokens": 512},
        step_duration_ms: float = 10.0,  # Simulated time per forward pass
    ):
        self.llm = emulate_init(model=model, args=args)
        self.vllm_config = self.llm.llm_engine.vllm_config
        cache_cfg = self.vllm_config.cache_config
        model_cfg = self.vllm_config.model_config
        parallel_cfg = self.vllm_config.parallel_config

        self.max_model_len: int = model_cfg.max_model_len
        self.block_size: int = cache_cfg.block_size
        self.num_blocks: int = cache_cfg.num_gpu_blocks
        self.num_kv_heads: int = model_cfg.get_num_kv_heads(parallel_cfg)
        self.head_size: int = model_cfg.get_head_size()

        self.step_duration_ms = step_duration_ms
        self.step_duration_s = step_duration_ms / 1000.0

        # Create scheduler
        self.scheduler= self.llm.llm_engine.engine_core.engine_core.scheduler

        self.max_num_seqs: int = self.scheduler.max_num_running_reqs
        self.max_num_batched_tokens: int = self.scheduler.max_num_scheduled_tokens

        # Tracking
        self.request_arrival_times: Dict[str, float] = {}
        self.request_first_token_times: Dict[str, float] = {}
        self.request_finish_times: Dict[str, float] = {}

    def dump_config(self):
        """Print all configuration fields extracted from the LLM object."""
        cfg = self.vllm_config
        print("=" * 60)
        print("SchedulerSimulator Configuration")
        print("=" * 60)
        print(f"  max_model_len          : {self.max_model_len}")
        print(f"  block_size             : {self.block_size}")
        print(f"  num_blocks             : {self.num_blocks}")
        print(f"  num_kv_heads           : {self.num_kv_heads}")
        print(f"  head_size              : {self.head_size}")
        print(f"  max_num_seqs           : {self.max_num_seqs}")
        print(f"  max_num_batched_tokens : {self.max_num_batched_tokens}")
        print(f"  step_duration_ms       : {self.step_duration_ms}")
        # print()
        # print("  vllm_config:")
        # print(f"    model_config              : {cfg.model_config}")
        # print(f"    cache_config              : {cfg.cache_config}")
        # print(f"    parallel_config           : {cfg.parallel_config}")
        # print(f"    scheduler_config          : {cfg.scheduler_config}")
        # print(f"    device_config             : {cfg.device_config}")
        # print(f"    load_config               : {cfg.load_config}")
        # print(f"    offload_config            : {cfg.offload_config}")
        # print(f"    attention_config          : {cfg.attention_config}")
        # print(f"    kernel_config             : {cfg.kernel_config}")
        # print(f"    lora_config               : {cfg.lora_config}")
        # print(f"    speculative_config        : {cfg.speculative_config}")
        # print(f"    structured_outputs_config : {cfg.structured_outputs_config}")
        # print(f"    observability_config      : {cfg.observability_config}")
        # print(f"    quant_config              : {cfg.quant_config}")
        # print(f"    compilation_config        : {cfg.compilation_config}")
        # print(f"    profiler_config           : {cfg.profiler_config}")
        # print(f"    kv_transfer_config        : {cfg.kv_transfer_config}")
        # print(f"    kv_events_config          : {cfg.kv_events_config}")
        # print(f"    ec_transfer_config        : {cfg.ec_transfer_config}")
        # print(f"    additional_config         : {cfg.additional_config}")
        print("=" * 60)

    def _create_mock_output(self, scheduler_output, token_id: int = 1000):
        return create_mock_output(self, scheduler_output, token_id)
        
    def _add_request(self, config: RequestConfig):
        req = make_request(self, config)
        print(f"Adding Request {req.request_id} of prompt length {len(req.prompt_token_ids)}")
        self.scheduler.add_request(req)
        self.request_arrival_times[config.request_id] = config.arrival_time

    def run(
        self,
        request_stream: Iterator[RequestConfig],
        max_steps: int = 10000,
        verbose: bool = False,
    ) -> SimulationResult:
        result = SimulationResult()
        all_requests = list(request_stream)
        all_requests.sort(key=lambda r: r.arrival_time)
        result.total_requests = len(all_requests)

        request_idx = 0
        sim_time = 0.0
        step_id = 0
        prev_output = None 

        if verbose:
            print(f"Starting simulation with {len(all_requests)} requests")
            print(f"Block capacity: {self.num_blocks} blocks x {self.block_size} = {self.num_blocks * self.block_size} tokens")
            print()

        while step_id < max_steps:
            # 1. Add any requests that have arrived by this time
            while request_idx < len(all_requests) and all_requests[request_idx].arrival_time <= sim_time:
                req_config = all_requests[request_idx]
                self._add_request(req_config)
                if verbose:
                    print(f"[t={sim_time:.3f}s] Request {req_config.request_id} arrived "
                          f"(prompt={len(req_config.prompt_token_ids)}, max_out={req_config.max_tokens})")
                request_idx += 1

            # 2. Check if there's work to do
            if not self.scheduler.has_requests() and request_idx >= len(all_requests):
                if verbose:
                    print(f"\n[t={sim_time:.3f}s] All requests completed!")
                break

            if not self.scheduler.has_requests():
                # No requests yet, advance time to next arrival
                if request_idx < len(all_requests):
                    sim_time = all_requests[request_idx].arrival_time
                continue

            # 3. If we have a previous output, update scheduler state
            if prev_output is not None and prev_output.total_num_scheduled_tokens > 0:
                mock_output = self._create_mock_output(prev_output, token_id=1000 + step_id)
                engine_outputs = self.scheduler.update_from_output(prev_output, mock_output)
                
                # Track finished requests
                for client_idx, outputs in engine_outputs.items():
                    for out in outputs.outputs:
                        if out.finish_reason:
                            self.request_finish_times[out.request_id] = sim_time
                            if verbose:
                                latency = sim_time - self.request_arrival_times[out.request_id]
                                print(f"[t={sim_time:.3f}s] Request {out.request_id} FINISHED "
                                      f"(reason={out.finish_reason}, latency={latency:.3f}s)")

            # 4. Schedule next batch
            sched_output = self.scheduler.schedule()

            # Track first token times for new requests
            for new_req in sched_output.scheduled_new_reqs:
                if new_req.req_id not in self.request_first_token_times:
                    self.request_first_token_times[new_req.req_id] = sim_time

            # 5. Record step metrics
            free_blocks = self.scheduler.kv_cache_manager.block_pool.get_num_free_blocks()

            # Check for preemptions (requests that were running but moved back to waiting)
            preempted = []
            if sched_output.preempted_req_ids:
                preempted = list(sched_output.preempted_req_ids)
                result.preemption_count += len(preempted)

            step_record = SimulationStep(
                step_id=step_id,
                sim_time=sim_time,
                num_scheduled_new=len(sched_output.scheduled_new_reqs),
                num_scheduled_cached=sched_output.scheduled_cached_reqs.num_reqs,
                total_tokens_scheduled=sched_output.total_num_scheduled_tokens,
                scheduled_req_ids=list(sched_output.num_scheduled_tokens.keys()),
                free_blocks=free_blocks,
                finished_req_ids=list(sched_output.finished_req_ids),
                preempted_req_ids=preempted,
                waiting_count=len(self.scheduler.waiting),
                running_count=len(self.scheduler.running),
            )
            result.steps.append(step_record)

            if verbose and sched_output.total_num_scheduled_tokens > 0:
                print(f"[t={sim_time:.3f}s] Step {step_id}: "
                      f"batch={len(sched_output.num_scheduled_tokens)}, "
                      f"tokens={sched_output.total_num_scheduled_tokens}, "
                      f"free_blocks={free_blocks}, "
                      f"waiting={len(self.scheduler.waiting)}, "
                      f"running={len(self.scheduler.running)}")
                if preempted:
                    print(f"           PREEMPTED: {preempted}")

            # 5.5. Estimate the step duration using database (placeholder)
            self.step_duration_s = self.step_duration_s

            # 6. Advance simulation time
            if sched_output.total_num_scheduled_tokens > 0:
                result.total_tokens_generated += sched_output.total_num_scheduled_tokens
                sim_time += self.step_duration_s
                step_id += 1
            else:
                # No work scheduled, advance to next arrival or small increment
                if request_idx < len(all_requests):
                    sim_time = all_requests[request_idx].arrival_time
                else:
                    sim_time += self.step_duration_s

            prev_output = sched_output

        result.total_steps = step_id

        # Calculate latencies
        for req_id, arrival in self.request_arrival_times.items():
            if req_id in self.request_finish_times:
                result.request_latencies[req_id] = self.request_finish_times[req_id] - arrival
            if req_id in self.request_first_token_times:
                result.request_ttft[req_id] = self.request_first_token_times[req_id] - arrival

        return result


def print_summary(result: SimulationResult):
    """Print simulation summary."""
    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)

    print(f"\nRequests: {result.total_requests}")
    print(f"Steps: {result.total_steps}")
    print(f"Total tokens generated: {result.total_tokens_generated}")
    print(f"Preemptions: {result.preemption_count}")

    if result.request_latencies:
        latencies = list(result.request_latencies.values())
        print(f"\nLatency (s):")
        print(f"  Mean: {sum(latencies)/len(latencies):.3f}")
        print(f"  Min:  {min(latencies):.3f}")
        print(f"  Max:  {max(latencies):.3f}")

    if result.request_ttft:
        ttfts = list(result.request_ttft.values())
        print(f"\nTime to First Token (s):")
        print(f"  Mean: {sum(ttfts)/len(ttfts):.3f}")
        print(f"  Min:  {min(ttfts):.3f}")
        print(f"  Max:  {max(ttfts):.3f}")

    # Block usage over time
    if result.steps:
        block_usage = [s.free_blocks for s in result.steps if s.total_tokens_scheduled > 0]
        if block_usage:
            print(f"\nFree blocks:")
            print(f"  Mean: {sum(block_usage)/len(block_usage):.1f}")
            print(f"  Min:  {min(block_usage)}")
            print(f"  Max:  {max(block_usage)}")


if __name__ == "__main__":
    print("=" * 60)
    print("Dooly Sim (emualting vLLM scheduler)")
    print("=" * 60)
    
    # Create request stream: 20 requests at ~5 req/sec
    stream = RequestStream(
        num_requests=20,
        arrival_rate=5.0,  # 5 requests per second average
        prompt_len_range=(20, 50),
        output_len_range=(10, 30),
        seed=42,
    )

    simulator = DoolySim(
        step_duration_ms=10.0,
        model="gpt2",
        args={"max_model_len": 512, "gpu_memory_utilization": 0.0003439735828288387, "max_num_seqs": 10, "max_num_batched_tokens": 512},
    )

    simulator.dump_config()

    # Run simulation
    result = simulator.run(
        request_stream=stream.generate(),
        max_steps=1000,
        verbose=True,
    )

    print_summary(result)

