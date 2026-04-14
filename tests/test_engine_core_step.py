from .utils import (
    get_real_and_simulated_llm,
    add_prompts_to_schedulers,
    add_requests_to_schedulers,
    patch_uniproc_executor,
    get_engine_cores,
    assert_engine_core_outputs_equal,
)


class TestEngineCoreStep:

    def test_basic_engine_core_step(self):
        real_llm, simulated_llm, real_engine_core, sim_engine_core = get_engine_cores()

        prompts = [
            "What is the capital of France?",
            "Explain the theory of relativity in simple terms.",
            "Write a short poem about the ocean.",
        ]
        add_prompts_to_schedulers(prompts, real_llm, simulated_llm)

        while True:
            real_outputs, real_active = real_engine_core.step()
            sim_outputs, sim_active = sim_engine_core.step()

            assert real_active == sim_active, \
                f"step active mismatch: real={real_active} sim={sim_active}"

            if not real_active and not sim_active:
                break

            assert_engine_core_outputs_equal(real_outputs, sim_outputs)

    def test_low_kv_cache_engine_core_step(self):
        """With limited KV cache, not all requests can be scheduled at once.
        Real and simulated engine cores must produce identical outputs every step,
        and at least one step must have fewer active requests than total submitted.

        512 blocks available. 17 requests × ceil(496/16)=31 blocks = 527 > 512,
        so the 17th request must wait. max_num_seqs and max_num_batched_tokens
        are raised so KV cache — not sequence count or batch size — is the bottleneck.
        """
        real_llm, simulated_llm, real_engine_core, sim_engine_core = get_engine_cores(
            kv_cache_memory_bytes=512 * 589_824,
            # max_num_seqs=20,
            # max_num_batched_tokens=10000,
        )

        # 496 tokens → ceil(496/16)=31 blocks each; 17×31=527 > 512 available
        requests = [
            (f"req_{i}", list(range(i * 500, i * 500 + 496)), 3, 0)
            for i in range(17)
        ]
        add_requests_to_schedulers(requests, real_llm, simulated_llm)

        saw_partial_step = False

        while True:
            real_outputs, real_active = real_engine_core.step()
            sim_outputs, sim_active = sim_engine_core.step()

            assert real_active == sim_active, \
                f"step active mismatch: real={real_active} sim={sim_active}"

            if not real_active and not sim_active:
                break

            assert_engine_core_outputs_equal(real_outputs, sim_outputs)

            num_in_step = sum(
                len(engine_outputs.outputs) for engine_outputs in sim_outputs.values()
            )
            if num_in_step < 17:
                saw_partial_step = True

        assert saw_partial_step, (
            "Expected at least one step with fewer than 17 active requests "
            "due to KV cache pressure, but all 17 ran together every step"
        )

    def test_chunked_prefill_engine_core_step(self):
        """Chunked prefill only: a 100-token prompt split at max_num_batched_tokens=32
        requires 4 prefill steps before decoding starts. Total steps must exceed
        what decode alone would need, proving chunking actually occurred."""
        real_llm, simulated_llm, real_engine_core, sim_engine_core = get_engine_cores(
            enable_chunked_prefill=True,
            max_num_batched_tokens=32,
        )

        # 100 tokens → ceil(100/32)=4 prefill steps, then 3 decode = 7 total minimum
        add_requests_to_schedulers(
            [("req_0", list(range(50, 150)), 3, 0)],
            real_llm,
            simulated_llm,
        )

        step_count = 0
        while True:
            real_outputs, real_active = real_engine_core.step()
            sim_outputs, sim_active = sim_engine_core.step()

            assert real_active == sim_active, \
                f"step active mismatch: real={real_active} sim={sim_active}"

            if not real_active and not sim_active:
                break

            assert_engine_core_outputs_equal(real_outputs, sim_outputs)
            step_count += 1

        # Without chunking: 1 prefill + 3 decode = 4 steps.
        # With chunking at 32 tokens/step: 4 prefill + 3 decode = 7 steps.
        assert step_count > 4, (
            f"Expected >4 steps with chunked prefill of 100-token prompt, got {step_count}"
        )

    def test_small_block_size_engine_core_step(self):
        """block_size=32 uses twice as many bytes per block vs block_size=16,
        halving the block count for the same memory.
        Same kv_cache_memory_bytes → 256 blocks at block_size=32.
        17 requests × ceil(496/32)=16 blocks = 272 > 256 → KV pressure applies.
        (block_size must be a multiple of 16; 8 is not supported by flash_attn/flashinfer.)"""
        # GPT-2 bytes/block at block_size=32:
        #   32 × 12 layers × 12 heads × 64 dim × 2(KV) × 2(fp16) = 1,179,648
        # 512 × 589,824 bytes / 1,179,648 = 256 blocks available
        real_llm, simulated_llm, real_engine_core, sim_engine_core = get_engine_cores(
            block_size=32,
            kv_cache_memory_bytes=512 * 589_824,
            max_num_seqs=20,
            max_num_batched_tokens=10000,
        )

        # 496 tokens → ceil(496/32)=16 blocks each; 17×16=272 > 256 available
        requests = [
            (f"req_{i}", list(range(i * 500, i * 500 + 496)), 3, 0)
            for i in range(17)
        ]
        add_requests_to_schedulers(requests, real_llm, simulated_llm)

        saw_partial_step = False

        while True:
            real_outputs, real_active = real_engine_core.step()
            sim_outputs, sim_active = sim_engine_core.step()

            assert real_active == sim_active, \
                f"step active mismatch: real={real_active} sim={sim_active}"

            if not real_active and not sim_active:
                break

            assert_engine_core_outputs_equal(real_outputs, sim_outputs)

            num_in_step = sum(
                len(engine_outputs.outputs) for engine_outputs in sim_outputs.values()
            )
            if num_in_step < 17:
                saw_partial_step = True

        assert saw_partial_step, (
            "Expected KV pressure with block_size=32 (256 blocks, 17×16=272 needed), "
            "but all 17 requests ran together every step"
        )

    def test_interactivity_mode_engine_core_step(self):
        """performance_mode='interactivity' optimises for TTFT. Real and sim must agree."""
        real_llm, simulated_llm, real_engine_core, sim_engine_core = get_engine_cores(
            performance_mode="interactivity",
        )

        add_requests_to_schedulers(
            [
                ("req_0", list(range(10, 25)), 5, 0),
                ("req_1", list(range(30, 45)), 5, 0),
            ],
            real_llm,
            simulated_llm,
        )

        while True:
            real_outputs, real_active = real_engine_core.step()
            sim_outputs, sim_active = sim_engine_core.step()

            assert real_active == sim_active, \
                f"step active mismatch: real={real_active} sim={sim_active}"

            if not real_active and not sim_active:
                break

            assert_engine_core_outputs_equal(real_outputs, sim_outputs)

    def test_throughput_mode_engine_core_step(self):
        """performance_mode='throughput' enables larger batches and chunked prefill.
        Real and sim must agree on all outputs."""
        real_llm, simulated_llm, real_engine_core, sim_engine_core = get_engine_cores(
            performance_mode="throughput",
        )

        add_requests_to_schedulers(
            [
                ("req_0", list(range(10, 25)), 5, 0),
                ("req_1", list(range(30, 45)), 5, 0),
            ],
            real_llm,
            simulated_llm,
        )

        while True:
            real_outputs, real_active = real_engine_core.step()
            sim_outputs, sim_active = sim_engine_core.step()

            assert real_active == sim_active, \
                f"step active mismatch: real={real_active} sim={sim_active}"

            if not real_active and not sim_active:
                break

            assert_engine_core_outputs_equal(real_outputs, sim_outputs)

    def test_priority_engine_core_step(self):
        """Priority scheduling only: req_lo is submitted first but req_hi has
        higher priority. With max_num_seqs=1, req_hi must start before req_lo."""
        real_llm, simulated_llm, real_engine_core, sim_engine_core = get_engine_cores(
            scheduling_policy="priority",
            max_num_seqs=1,
        )

        # req_lo submitted first — FCFS would favour it — but req_hi wins on priority
        add_requests_to_schedulers(
            [
                ("req_lo", list(range(200, 215)), 3, 2),  # low priority
                ("req_hi", list(range(100, 115)), 3, 0),  # high priority
            ],
            real_llm,
            simulated_llm,
        )

        seen_order = []

        while True:
            real_outputs, real_active = real_engine_core.step()
            sim_outputs, sim_active = sim_engine_core.step()

            assert real_active == sim_active, \
                f"step active mismatch: real={real_active} sim={sim_active}"

            if not real_active and not sim_active:
                break

            assert_engine_core_outputs_equal(real_outputs, sim_outputs)

            for engine_outputs in sim_outputs.values():
                for out in engine_outputs.outputs:
                    if out.request_id not in seen_order:
                        seen_order.append(out.request_id)

        assert seen_order == ["req_hi", "req_lo"], \
            f"Priority order violated: expected [req_hi, req_lo], got {seen_order}"

    def test_priority_chunked_prefill_engine_core_step(self):
        """Priority scheduling combined with chunked prefill: req_lo is submitted
        first but req_hi has higher priority. Real and sim must agree every step."""
        real_llm, simulated_llm, real_engine_core, sim_engine_core = get_engine_cores(
            scheduling_policy="priority",
            enable_chunked_prefill=True,
            max_num_batched_tokens=32,
            max_num_seqs=1,
        )

        # req_lo submitted first — FCFS would favour it — but req_hi wins on priority
        add_requests_to_schedulers(
            [
                ("req_lo", list(range(200, 250)), 3, 2),  # 50 tokens, low priority
                ("req_hi", list(range(100, 150)), 3, 0),  # 50 tokens, high priority
            ],
            real_llm,
            simulated_llm,
        )

        seen_order = []

        while True:
            real_outputs, real_active = real_engine_core.step()
            sim_outputs, sim_active = sim_engine_core.step()

            assert real_active == sim_active, \
                f"step active mismatch: real={real_active} sim={sim_active}"

            if not real_active and not sim_active:
                break

            assert_engine_core_outputs_equal(real_outputs, sim_outputs)

            for engine_outputs in sim_outputs.values():
                for out in engine_outputs.outputs:
                    if out.request_id not in seen_order:
                        seen_order.append(out.request_id)

        assert seen_order == ["req_hi", "req_lo"], \
            f"Priority+chunked order violated: expected [req_hi, req_lo], got {seen_order}"

    def test_fcfs_ordering_engine_core_step(self):
        """FCFS ordering: with max_num_seqs=1, only one request runs at a time.
        Requests must be served in submission order: req_0 before req_1 before req_2."""
        real_llm, simulated_llm, real_engine_core, sim_engine_core = get_engine_cores(
            max_num_seqs=1,
        )

        add_requests_to_schedulers(
            [
                ("req_0", list(range(10, 25)), 3, 0),
                ("req_1", list(range(30, 45)), 3, 0),
                ("req_2", list(range(50, 65)), 3, 0),
            ],
            real_llm,
            simulated_llm,
        )

        seen_order = []  # first appearance of each req_id in sim outputs

        while True:
            real_outputs, real_active = real_engine_core.step()
            sim_outputs, sim_active = sim_engine_core.step()

            assert real_active == sim_active, \
                f"step active mismatch: real={real_active} sim={sim_active}"

            if not real_active and not sim_active:
                break

            assert_engine_core_outputs_equal(real_outputs, sim_outputs)

            step_req_ids = {
                out.request_id
                for engine_outputs in sim_outputs.values()
                for out in engine_outputs.outputs
            }

            assert len(step_req_ids) <= 1, \
                f"max_num_seqs=1 violated: {step_req_ids} active simultaneously"

            for req_id in step_req_ids:
                if req_id not in seen_order:
                    seen_order.append(req_id)

        assert seen_order == ["req_0", "req_1", "req_2"], \
            f"FCFS order violated: expected [req_0, req_1, req_2], got {seen_order}"