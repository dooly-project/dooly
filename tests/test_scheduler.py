from .utils import (
    get_real_and_simulated_llm,
    add_prompts_to_schedulers,
    add_requests_to_schedulers,
    assert_scheduler_outputs_equal,
)


class TestScheduler:

    def test_basic_scheduler_step(self):
        real_llm, simulated_llm = get_real_and_simulated_llm()
        prompts = [
            "What is the capital of France?",
            "Explain the theory of relativity in simple terms.",
            "Write a short poem about the ocean.",
        ]
        add_prompts_to_schedulers(prompts, real_llm, simulated_llm)

        real_scheduler = real_llm.llm_engine.engine_core.engine_core.scheduler
        sim_scheduler = simulated_llm.scheduler
        real_sched_out = real_scheduler.schedule()
        sim_sched_out = sim_scheduler.schedule()

        assert_scheduler_outputs_equal(real_sched_out, sim_sched_out)

    def test_fcfs_scheduler_step(self):
        """FCFS: all requests added simultaneously are scheduled in step 1;
        a 4th request added after max_num_seqs=3 is saturated must stay waiting."""
        # Can only schedule 3 concurrent requests at once
        real_llm, simulated_llm = get_real_and_simulated_llm(max_num_seqs=3)
        prompts = [
            "First request.",
            "Second request.",
            "Third request.",
        ]
        add_prompts_to_schedulers(prompts, real_llm, simulated_llm)

        real_scheduler = real_llm.llm_engine.engine_core.engine_core.scheduler
        sim_scheduler = simulated_llm.scheduler

        # Step 1: all 3 requests fit exactly within max_num_seqs=3
        real_sched_out = real_scheduler.schedule()
        sim_sched_out = sim_scheduler.schedule()

        assert_scheduler_outputs_equal(real_sched_out, sim_sched_out)

        scheduled_ids = set(sim_sched_out.num_scheduled_tokens.keys())
        assert scheduled_ids == {"req_0", "req_1", "req_2"}, \
            f"FCFS step 1: expected all 3 requests scheduled, got {scheduled_ids}"

        # Step 2: add a 4th request while 3 are already running (saturates max_num_seqs)
        tokenizer = simulated_llm.tokenizer
        token_ids = tokenizer.encode("Fourth request.")
        add_requests_to_schedulers(
            [("req_3", token_ids, 20, 0)],
            real_llm,
            simulated_llm,
        )

        real_sched_out2 = real_scheduler.schedule()
        sim_sched_out2 = sim_scheduler.schedule()

        assert_scheduler_outputs_equal(real_sched_out2, sim_sched_out2)

        # req_3 must be absent — max_num_seqs=3 is already full
        scheduled_ids2 = set(sim_sched_out2.num_scheduled_tokens.keys())
        assert "req_3" not in scheduled_ids2, \
            f"FCFS step 2: req_3 should be waiting (max_num_seqs=3 full), got {scheduled_ids2}"

    def test_priority_scheduler_step(self):
        """Priority: the highest-priority (lowest int) request is scheduled first when capacity is limited."""
        real_llm, simulated_llm = get_real_and_simulated_llm(
            scheduling_policy="priority",
            max_num_seqs=1,  # Force only 1 request at a time so priority matters
        )

        tokenizer = simulated_llm.tokenizer
        token_ids = tokenizer.encode("A short prompt.")

        # req_0: low priority (2), req_1: high priority (0), req_2: medium priority (1)
        add_requests_to_schedulers(
            [
                ("req_0", token_ids, 20, 2),
                ("req_1", token_ids, 20, 0),
                ("req_2", token_ids, 20, 1),
            ],
            real_llm,
            simulated_llm,
        )

        real_scheduler = real_llm.llm_engine.engine_core.engine_core.scheduler
        sim_scheduler = simulated_llm.scheduler
        real_sched_out = real_scheduler.schedule()
        sim_sched_out = sim_scheduler.schedule()

        assert_scheduler_outputs_equal(real_sched_out, sim_sched_out)

        # With max_num_seqs=1, only the highest-priority request (req_1, priority=0) should run
        scheduled_ids = set(sim_sched_out.num_scheduled_tokens.keys())
        assert scheduled_ids == {"req_1"}, \
            f"Priority expected only req_1 (priority=0) scheduled, got {scheduled_ids}"

    def test_prefix_caching_scheduler_step(self):
        """Prefix caching: a request added after the first step should get
        a cache hit on the shared 32-token prefix."""
        real_llm, simulated_llm = get_real_and_simulated_llm()

        # 32-token shared prefix followed by unique suffixes
        shared_prefix = list(range(100, 132))  # tokens 100..131
        suffix_a = list(range(200, 210))
        suffix_b = list(range(300, 310))

        add_requests_to_schedulers(
            [
                ("req_0", shared_prefix + suffix_a, 20, 0),
                ("req_1", shared_prefix + suffix_b, 20, 0),
            ],
            real_llm,
            simulated_llm,
        )

        real_scheduler = real_llm.llm_engine.engine_core.engine_core.scheduler
        sim_scheduler = simulated_llm.scheduler

        # Step 1: schedule the initial two requests
        real_sched_out = real_scheduler.schedule()
        sim_sched_out = sim_scheduler.schedule()

        assert_scheduler_outputs_equal(real_sched_out, sim_sched_out)

        # Advance scheduler state so prefix blocks are marked as computed
        mock_out = simulated_llm._create_mock_output(sim_sched_out)
        real_scheduler.update_from_output(real_sched_out, mock_out)
        sim_scheduler.update_from_output(sim_sched_out, mock_out)

        # Step 2: add a third request that shares the same 32-token prefix
        suffix_c = list(range(400, 410))
        add_requests_to_schedulers(
            [("req_2", shared_prefix + suffix_c, 20, 0)],
            real_llm,
            simulated_llm,
        )

        real_sched_out2 = real_scheduler.schedule()
        sim_sched_out2 = sim_scheduler.schedule()

        assert_scheduler_outputs_equal(real_sched_out2, sim_sched_out2)

        # req_2 should be a new request with num_computed_tokens > 0 (prefix cache hit)
        sim_new_by_id = {r.req_id: r for r in sim_sched_out2.scheduled_new_reqs}
        assert "req_2" in sim_new_by_id, "req_2 should appear as a newly scheduled request"
        assert sim_new_by_id["req_2"].num_computed_tokens > 0, (
            f"req_2 should have a prefix cache hit, "
            f"got num_computed_tokens={sim_new_by_id['req_2'].num_computed_tokens}"
        )


    def test_awq_quantization_scheduler_step(self):
        """AWQ quantization: real and simulated schedulers must agree on
        which requests are scheduled and how many tokens are assigned."""
        real_llm, simulated_llm = get_real_and_simulated_llm(model="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
                                                             quantization="awq")

        tokenizer = simulated_llm.tokenizer
        token_ids = tokenizer.encode("This is the first request.")

        add_requests_to_schedulers(
            [
                ("req_0", token_ids, 20, 0),
                ("req_1", token_ids, 20, 0),
            ],
            real_llm,
            simulated_llm,
        )

        real_scheduler = real_llm.llm_engine.engine_core.engine_core.scheduler
        sim_scheduler = simulated_llm.scheduler

        real_sched_out = real_scheduler.schedule()
        sim_sched_out = sim_scheduler.schedule()

        assert_scheduler_outputs_equal(real_sched_out, sim_sched_out)


    def test_fp8_kv_cache_scheduler_step(self):
        """FP8 KV cache uses 1 byte per element vs 2 for fp16, halving bytes per block.
        Verifies real and simulated schedulers agree on scheduled requests and token counts."""
        real_llm, simulated_llm = get_real_and_simulated_llm(kv_cache_dtype="fp8")

        tokenizer = simulated_llm.tokenizer
        token_ids = tokenizer.encode("This is the first request.")

        add_requests_to_schedulers(
            [
                ("req_0", token_ids, 20, 0),
                ("req_1", token_ids, 20, 0),
            ],
            real_llm,
            simulated_llm,
        )

        real_scheduler = real_llm.llm_engine.engine_core.engine_core.scheduler
        sim_scheduler = simulated_llm.scheduler

        real_sched_out = real_scheduler.schedule()
        sim_sched_out = sim_scheduler.schedule()

        assert_scheduler_outputs_equal(real_sched_out, sim_sched_out)
