from ..utils import SimulationResult, SimulationStep, RequestTrace, TokenEvent
from ..request import RequestConfig, make_request, create_mock_output
from ..emulator import emulate_init
from .latency_estimator import LatencyEstimator

from typing import Dict, Iterator

class DoolySim:
    def __init__(
        self,
        model: str = "gpt2",
        args: Dict = {"max_model_len": 512, "gpu_memory_utilization": 0.0003439735828288387, "max_num_seqs": 10, "max_num_batched_tokens": 512},
        step_duration_ms: float = 10.0,  # Simulated time per forward pass (fallback)
        cache_dir: str = "/home/jkim38/dooly/simulator/llama-3.1-8b-FLASHINFER",
        db_path: str = "/home/jkim38/dooly/profiler/database/dooly_data.db",
        backend: str = "FLASHINFER",
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

        # Create scheduler
        self.scheduler= self.llm.llm_engine.engine_core.engine_core.scheduler

        self.max_num_seqs: int = self.scheduler.max_num_running_reqs
        self.max_num_batched_tokens: int = self.scheduler.max_num_scheduled_tokens

        # Initialize latency estimator
        self.model_name = model
        self.backend = backend
        self.latency_estimator = LatencyEstimator(
            model_name=model,
            backend=backend,
            vllm_config=self.vllm_config,
            cache_dir=cache_dir,
            db_path=db_path,
            fallback_duration_ms=step_duration_ms
        )

        # Tracking (all in milliseconds)
        self.request_arrival_times_ms: Dict[str, float] = {}
        self.request_first_token_times_ms: Dict[str, float] = {}
        self.request_finish_times_ms: Dict[str, float] = {}

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
        print("=" * 60)

    def _create_mock_output(self, scheduler_output, token_id: int = 1000):
        return create_mock_output(self, scheduler_output, token_id)
        
    def _add_request(self, config: RequestConfig, result: SimulationResult):
        req = make_request(self, config)
        print(f"Adding Request {req.request_id} of prompt length {len(req.prompt_token_ids)}")
        self.scheduler.add_request(req)

        # Convert arrival time to milliseconds (config uses seconds)
        arrival_time_ms = config.arrival_time * 1000.0
        self.request_arrival_times_ms[config.request_id] = arrival_time_ms

        # Create RequestTrace for detailed tracking
        result.request_traces[config.request_id] = RequestTrace(
            request_id=config.request_id,
            arrival_time_ms=arrival_time_ms,
            prompt_len=len(config.prompt_token_ids),
            max_output_len=config.max_tokens,
        )

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
        sim_time_ms = 0.0 
        step_id = 0
        prev_output = None

        if verbose:
            print(f"Starting simulation with {len(all_requests)} requests")
            print(f"Block capacity: {self.num_blocks} blocks x {self.block_size} = {self.num_blocks * self.block_size} tokens")
            print()

        while step_id < max_steps:
            # 1. Add any requests that have arrived by this time
            while request_idx < len(all_requests) and all_requests[request_idx].arrival_time * 1000.0 <= sim_time_ms:
                req_config = all_requests[request_idx]
                self._add_request(req_config, result)
                if verbose:
                    print(f"[t={sim_time_ms:.1f}ms] Request {req_config.request_id} arrived "
                          f"(prompt={len(req_config.prompt_token_ids)}, max_out={req_config.max_tokens})")
                request_idx += 1

            # 2. Check if there's work to do
            if not self.scheduler.has_requests() and request_idx >= len(all_requests):
                if verbose:
                    print(f"\n[t={sim_time_ms:.1f}ms] All requests completed!")
                break

            if not self.scheduler.has_requests():
                # No requests yet, advance time to next arrival
                if request_idx < len(all_requests):
                    sim_time_ms = all_requests[request_idx].arrival_time * 1000.0
                continue

            # 3. If we have a previous output, update scheduler state
            if prev_output is not None and prev_output.total_num_scheduled_tokens > 0:
                # Create mock output
                mock_output = self._create_mock_output(prev_output, token_id=1000 + step_id)
                engine_outputs = self.scheduler.update_from_output(prev_output, mock_output)
                
                # Track finished requests and token events
                batch_size = len(prev_output.num_scheduled_tokens) if prev_output else 0

                for _, outputs in engine_outputs.items():
                    for out in outputs.outputs:
                        # Record token events for newly generated tokens
                        if out.new_token_ids and len(out.new_token_ids) > 0:
                            if out.request_id not in self.request_first_token_times_ms:
                                self.request_first_token_times_ms[out.request_id] = sim_time_ms

                            # Record each newly generated token
                            if out.request_id in result.request_traces:
                                trace = result.request_traces[out.request_id]
                                for _ in out.new_token_ids:
                                    # Calculate per-token latency
                                    if len(trace.token_events) == 0:
                                        # First token: step_dur = TTFT (time from arrival to first token)
                                        arrival_time = self.request_arrival_times_ms[out.request_id]
                                        per_token_latency = sim_time_ms - arrival_time
                                    else:
                                        # Subsequent tokens: step_dur = inter-token latency
                                        prev_token_time = trace.token_events[-1].timestamp_ms
                                        per_token_latency = sim_time_ms - prev_token_time

                                    token_event = TokenEvent(
                                        token_index=len(trace.token_events),
                                        timestamp_ms=sim_time_ms,
                                        step_duration_ms=per_token_latency,
                                        batch_size=batch_size,
                                        total_tokens_in_batch=prev_output.total_num_scheduled_tokens if prev_output else 0,
                                    )
                                    trace.token_events.append(token_event)

                        if out.finish_reason:
                            self.request_finish_times_ms[out.request_id] = sim_time_ms

                            # Update request trace with finish information
                            if out.request_id in result.request_traces:
                                trace = result.request_traces[out.request_id]
                                trace.finish_time_ms = sim_time_ms
                                trace.finish_reason = str(out.finish_reason)

                            if verbose:
                                latency_ms = sim_time_ms - self.request_arrival_times_ms[out.request_id]
                                print(f"[t={sim_time_ms:.1f}ms] Request {out.request_id} FINISHED "
                                      f"(reason={out.finish_reason}, latency={latency_ms:.1f}ms)")

            # 4. Schedule next batch
            sched_output = self.scheduler.schedule()

            # 4.5. Track when requests are first scheduled (for queue time calculation)
            for new_req in sched_output.scheduled_new_reqs:
                if new_req.req_id in result.request_traces:
                    trace = result.request_traces[new_req.req_id]
                    if trace.scheduled_time_ms is None:
                        trace.scheduled_time_ms = sim_time_ms
                        if verbose:
                            queue_time = sim_time_ms - trace.arrival_time_ms
                            print(f"[t={sim_time_ms:.1f}ms] Request {new_req.req_id} first scheduled "
                                  f"(queue_time={queue_time:.1f}ms)")

            # 5. Record step metrics
            free_blocks = self.scheduler.kv_cache_manager.block_pool.get_num_free_blocks()

            # Check for preemptions (requests that were running but moved back to waiting)
            preempted = []
            if sched_output.preempted_req_ids:
                preempted = list(sched_output.preempted_req_ids)
                result.preemption_count += len(preempted)

            step_record = SimulationStep(
                step_id=step_id,
                sim_time_ms=sim_time_ms,
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
                print(f"[t={sim_time_ms:.1f}ms] Step {step_id}: "
                      f"batch={len(sched_output.num_scheduled_tokens)}, "
                      f"tokens={sched_output.total_num_scheduled_tokens}, "
                      f"free_blocks={free_blocks}, "
                      f"waiting={len(self.scheduler.waiting)}, "
                      f"running={len(self.scheduler.running)}")
                if preempted:
                    print(f"           PREEMPTED: {preempted}")

            # 5.5. Estimate the step duration using regression models
            if sched_output.total_num_scheduled_tokens > 0:
                self.step_duration_ms = self.latency_estimator.estimate_step_latency(sched_output, step_id=step_id)

            # 6. Advance simulation time
            if sched_output.total_num_scheduled_tokens > 0:
                result.total_tokens_generated += sched_output.total_num_scheduled_tokens
                sim_time_ms += self.step_duration_ms
                step_id += 1
            else:
                # No work scheduled, advance to next arrival or small increment
                if request_idx < len(all_requests):
                    sim_time_ms = all_requests[request_idx].arrival_time * 1000.0
                else:
                    sim_time_ms += self.step_duration_ms

            # this is wrong because this assumes that 
            # scheduler output is the same as engine output, which is not true when we have chunked prefill 
            prev_output = sched_output

        result.total_steps = step_id

        # Calculate latencies (in milliseconds)
        for req_id, arrival_ms in self.request_arrival_times_ms.items():
            if req_id in self.request_finish_times_ms:
                result.request_latencies[req_id] = self.request_finish_times_ms[req_id] - arrival_ms
            if req_id in self.request_first_token_times_ms:
                result.request_ttft[req_id] = self.request_first_token_times_ms[req_id] - arrival_ms

        return result

