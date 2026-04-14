import os
from concurrent.futures import Future

os.environ["VLLM_SKIP_P2P_CHECK"] = "1"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


def get_real_and_simulated_llm(
    model: str = "gpt2",
    sim_args: dict = {
        "step_duration_ms": 10.0,
        "backend": "FLASHINFER",
        "cpu_overhead": 10.0,
    },
    **llm_args,
):
    llm_args = {"max_model_len": 512, "kv_cache_memory_bytes": 10737418240, "max_num_seqs": 10, "max_num_batched_tokens": 512} | llm_args

    import vllm
    real_llm = vllm.LLM(model=model, load_format="dummy", enforce_eager=True, **llm_args)

    from ..core.dooly_sim import DoolySim
    simulated_llm = DoolySim(model=model, args=llm_args, **sim_args)
    return real_llm, simulated_llm

def get_engine_cores(**kwargs):
    real_llm, simulated_llm = get_real_and_simulated_llm(**kwargs)
    patch_uniproc_executor(simulated_llm)
    real_engine_core = real_llm.llm_engine.engine_core.engine_core
    sim_engine_core = simulated_llm.llm.llm_engine.engine_core.engine_core
    return real_llm, simulated_llm, real_engine_core, sim_engine_core


def add_prompts_to_schedulers(
    prompts: list[str],
    real_llm,
    simulated_llm,
) -> None:
    from ..request import RequestConfig, make_request

    tokenizer = simulated_llm.tokenizer
    configs = [
        RequestConfig(
            request_id=f"req_{i}",
            prompt_token_ids=tokenizer.encode(prompt),
            max_tokens=20,
            arrival_time=0.0,
        )
        for i, prompt in enumerate(prompts)
    ]

    real_scheduler = real_llm.llm_engine.engine_core.engine_core.scheduler
    sim_scheduler = simulated_llm.scheduler

    for config in configs:
        req = make_request(simulated_llm, config)
        real_scheduler.add_request(req)
        sim_scheduler.add_request(make_request(simulated_llm, config))


def add_requests_to_schedulers(
    requests: list[tuple[str, list[int], int, int]],
    real_llm,
    simulated_llm,
) -> None:
    """Add requests with explicit priorities to both schedulers.

    Args:
        requests: list of (request_id, prompt_token_ids, max_tokens, priority)
    """
    from vllm.v1.request import Request
    from vllm.sampling_params import SamplingParams

    real_scheduler = real_llm.llm_engine.engine_core.engine_core.scheduler
    sim_scheduler = simulated_llm.scheduler
    real_block_hasher = real_llm.llm_engine.engine_core.engine_core.request_block_hasher
    sim_block_hasher = simulated_llm.llm.llm_engine.engine_core.engine_core.request_block_hasher

    for req_id, token_ids, max_tokens, priority in requests:
        sampling_params = SamplingParams(max_tokens=max_tokens)
        real_scheduler.add_request(Request(
            request_id=req_id,
            prompt_token_ids=token_ids,
            sampling_params=sampling_params,
            pooling_params=None,
            block_hasher=real_block_hasher,
            priority=priority,
        ))
        sim_scheduler.add_request(Request(
            request_id=req_id,
            prompt_token_ids=token_ids,
            sampling_params=sampling_params,
            pooling_params=None,
            block_hasher=sim_block_hasher,
            priority=priority,
        ))

def assert_engine_core_outputs_equal(real_outputs, sim_outputs) -> None:
    """Compare dict[int, EngineCoreOutputs] from real and simulated engine cores."""

    def by_req_id(outputs_dict):
        result = {}
        for engine_outputs in outputs_dict.values():
            for out in engine_outputs.outputs:
                result[out.request_id] = out
        return result

    real_by_id = by_req_id(real_outputs)
    sim_by_id = by_req_id(sim_outputs)

    assert set(real_by_id) == set(sim_by_id), \
        f"output req ids: real={set(real_by_id)} sim={set(sim_by_id)}"

    for req_id in sim_by_id:
        real_out = real_by_id[req_id]
        sim_out = sim_by_id[req_id]
        assert real_out.new_token_ids == sim_out.new_token_ids, \
            f"req {req_id} new_token_ids: real={real_out.new_token_ids} sim={sim_out.new_token_ids}"
        assert real_out.num_cached_tokens == sim_out.num_cached_tokens, \
            f"req {req_id} num_cached_tokens: real={real_out.num_cached_tokens} sim={sim_out.num_cached_tokens}"
        assert real_out.num_external_computed_tokens == sim_out.num_external_computed_tokens, \
            f"req {req_id} num_external_computed_tokens: real={real_out.num_external_computed_tokens} sim={sim_out.num_external_computed_tokens}"
        
def assert_scheduler_outputs_equal(real_sched_out, sim_sched_out) -> None:
    # --- Token counts ---
    assert real_sched_out.total_num_scheduled_tokens == sim_sched_out.total_num_scheduled_tokens, \
        f"total tokens: real={real_sched_out.total_num_scheduled_tokens} sim={sim_sched_out.total_num_scheduled_tokens}"
    assert real_sched_out.num_scheduled_tokens == sim_sched_out.num_scheduled_tokens, \
        f"per-req tokens: real={real_sched_out.num_scheduled_tokens} sim={sim_sched_out.num_scheduled_tokens}"

    # --- Speculative decode / encoder inputs ---
    assert real_sched_out.scheduled_spec_decode_tokens == sim_sched_out.scheduled_spec_decode_tokens, \
        f"spec decode tokens: real={real_sched_out.scheduled_spec_decode_tokens} sim={sim_sched_out.scheduled_spec_decode_tokens}"
    assert real_sched_out.scheduled_encoder_inputs == sim_sched_out.scheduled_encoder_inputs, \
        f"encoder inputs: real={real_sched_out.scheduled_encoder_inputs} sim={sim_sched_out.scheduled_encoder_inputs}"
    assert real_sched_out.num_invalid_spec_tokens == sim_sched_out.num_invalid_spec_tokens, \
        f"invalid spec tokens: real={real_sched_out.num_invalid_spec_tokens} sim={sim_sched_out.num_invalid_spec_tokens}"

    # --- Prefix caching ---
    assert real_sched_out.num_common_prefix_blocks == sim_sched_out.num_common_prefix_blocks, \
        f"common prefix blocks: real={real_sched_out.num_common_prefix_blocks} sim={sim_sched_out.num_common_prefix_blocks}"

    # --- Finished / preempted / encoder frees ---
    assert real_sched_out.finished_req_ids == sim_sched_out.finished_req_ids, \
        f"finished req ids: real={real_sched_out.finished_req_ids} sim={sim_sched_out.finished_req_ids}"
    assert real_sched_out.preempted_req_ids == sim_sched_out.preempted_req_ids, \
        f"preempted req ids: real={real_sched_out.preempted_req_ids} sim={sim_sched_out.preempted_req_ids}"
    assert real_sched_out.free_encoder_mm_hashes == sim_sched_out.free_encoder_mm_hashes, \
        f"free encoder mm hashes: real={real_sched_out.free_encoder_mm_hashes} sim={sim_sched_out.free_encoder_mm_hashes}"

    # --- Structured output flags ---
    assert real_sched_out.has_structured_output_requests == sim_sched_out.has_structured_output_requests, \
        f"has_structured_output_requests: real={real_sched_out.has_structured_output_requests} sim={sim_sched_out.has_structured_output_requests}"
    assert real_sched_out.pending_structured_output_tokens == sim_sched_out.pending_structured_output_tokens, \
        f"pending_structured_output_tokens: real={real_sched_out.pending_structured_output_tokens} sim={sim_sched_out.pending_structured_output_tokens}"

    # --- scheduled_new_reqs ---
    real_new_by_id = {r.req_id: r for r in real_sched_out.scheduled_new_reqs}
    sim_new_by_id = {r.req_id: r for r in sim_sched_out.scheduled_new_reqs}
    assert real_new_by_id == sim_new_by_id, \
        f"new req ids: real={real_new_by_id} sim={sim_new_by_id}"
    for req_id in sim_new_by_id:
        real_r, sim_r = real_new_by_id[req_id], sim_new_by_id[req_id]
        # Tokens precomputed from prefix caching
        assert real_r.num_computed_tokens == sim_r.num_computed_tokens, \
            f"new req {req_id} num_computed_tokens: real={real_r.num_computed_tokens} sim={sim_r.num_computed_tokens}"
        # Allocated KV cache block IDs
        assert real_r.block_ids == sim_r.block_ids, \
            f"new req {req_id} block_ids: real={real_r.block_ids} sim={sim_r.block_ids}"

    # --- scheduled_cached_reqs (CachedRequestData) ---
    real_cr = real_sched_out.scheduled_cached_reqs
    sim_cr = sim_sched_out.scheduled_cached_reqs
    # All requests
    assert set(real_cr.req_ids) == set(sim_cr.req_ids), \
        f"cached req ids: real={set(real_cr.req_ids)} sim={set(sim_cr.req_ids)}"
    # Preempted requests that are now resumed
    assert real_cr.resumed_req_ids == sim_cr.resumed_req_ids, \
        f"resumed req ids: real={real_cr.resumed_req_ids} sim={sim_cr.resumed_req_ids}"

    # These fields are parallel arrays so access them with i
    real_cr_idx = {req_id: i for i, req_id in enumerate(real_cr.req_ids)}
    sim_cr_idx = {req_id: i for i, req_id in enumerate(sim_cr.req_ids)}
    for req_id in sim_cr.req_ids:
        ri, si = real_cr_idx[req_id], sim_cr_idx[req_id]
        # Number of tokens that are in KV cache
        assert real_cr.num_computed_tokens[ri] == sim_cr.num_computed_tokens[si], \
            f"cached req {req_id} num_computed_tokens: real={real_cr.num_computed_tokens[ri]} sim={sim_cr.num_computed_tokens[si]}"
        # Number of output tokens
        assert real_cr.num_output_tokens[ri] == sim_cr.num_output_tokens[si], \
            f"cached req {req_id} num_output_tokens: real={real_cr.num_output_tokens[ri]} sim={sim_cr.num_output_tokens[si]}"
        # New KV cache block allocations
        assert real_cr.new_block_ids[ri] == sim_cr.new_block_ids[si], \
            f"cached req {req_id} new_block_ids: real={real_cr.new_block_ids[ri]} sim={sim_cr.new_block_ids[si]}"


def make_fake_execute_model(dooly_sim):

    from ..request import create_mock_output

    def fake_execute_model(self, scheduler_output, non_block=False):
        f = Future()
        f.set_result(create_mock_output(dooly_sim, scheduler_output))
        return f

    return fake_execute_model


def patch_uniproc_executor(dooly_sim):
    """Monkeypatch UniProcExecutor.execute_model to use make_fake_execute_model.

    After this call every UniProcExecutor instance (including the one inside
    *real_llm*) will return a resolved Future wrapping a fake ModelRunnerOutput
    instead of running actual GPU kernels.

    Usage::

        patch_uniproc_executor(simulated_llm)
        real_llm, simulated_llm = get_real_and_simulated_llm()
        # real_llm.generate(...) now uses the fake executor
    """
    from vllm.v1.executor.uniproc_executor import UniProcExecutor

    UniProcExecutor.execute_model = make_fake_execute_model(dooly_sim)
