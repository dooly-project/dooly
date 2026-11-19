"""GPU check of fake data parallelism with expert parallelism (fake_tp._apply_fake_dp).

Fake DP=2 x TP=1 with EP on one GPU: the engine's own DP and EP GroupCoordinators
(emulated, 2 ranks each), the EP group's all2all manager (the platform
constructor's own selection for the configured backend), and the forward
context the engine builds for a 2-rank DP deployment. Drives the EP group's
dispatch and combine, as the MoE layer does, and checks that

* the forward context carries per-rank token counts [T, T] (every DP rank runs
  a batch like this one);
* dispatch returns rank 0's view of the gathered tokens (2T rows), with the
  gathered expert ids valid (the other rank's half replicates this rank's);
* combine returns this rank's T rows;
* the trace records what the transport runs -- all_gatherv and reduce_scatterv
  on the 2-rank DP group, with tainted per-rank sizes -- and the parser drops
  the dispatch / combine ranges themselves (they move data only through the DP
  group), so the comm profiler replays the primitives on a real group.

Runs directly on one GPU, under any vLLM env:
``python doolyprof/tests/dp_ep_emulation_test.py``
"""
import json
import os
import socket
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.environ["VLLM_FAKE_TP"] = "1"
os.environ["VLLM_FAKE_DP"] = "2"
os.environ["VLLM_FAKE_EP"] = "1"

import torch  # noqa: E402

from doolyprof.tracer.hooks import install_hooks  # noqa: E402

install_hooks()

from doolyprof.tracer import enable_taint_tracking  # noqa: E402
from doolyprof.tracer.tensor import TaintedTensor  # noqa: E402
from doolyprof.tracer.types import Taint, TaintedInt  # noqa: E402

DP = 2


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    cfg = VllmConfig(parallel_config=ParallelConfig(enable_expert_parallel=True))
    ctx = set_current_vllm_config(cfg)
    ctx.__enter__()
    init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                 distributed_init_method=f"tcp://127.0.0.1:{_free_port()}", backend="nccl")
    initialize_model_parallel(1, 1)

    from doolyprof.profiler.utils.fake_tp import apply_fake_tp_patches
    apply_fake_tp_patches(1, DP)
    import vllm.distributed.parallel_state as ps
    import vllm.forward_context as fc
    from doolyprof.profiler.comm_args import parse_comm_args

    failures = []
    # EP first: its all2all manager looks up the DP group while the EP group is
    # being built (the order GLM-4.7-Flash and Qwen3.6 reach them in), so the DP
    # group is built nested and must still get the platform's communicator.
    ep, dp = ps.get_ep_group(), ps.get_dp_group()
    for g in (dp, ep):
        if not callable(getattr(g.device_communicator, "all_gatherv", None)):
            failures.append(f"{g.unique_name}: communicator {type(g.device_communicator).__name__} has no all_gatherv")
    if (dp.world_size, ep.world_size) != (DP, DP):
        failures.append(f"group sizes dp={dp.world_size} ep={ep.world_size}, want {DP}")
    manager = getattr(ep.device_communicator, "all2all_manager", None)
    print(f"dp group {dp.unique_name} ({dp.world_size}), ep group {ep.unique_name} ({ep.world_size}), "
          f"all2all manager: {type(manager).__name__}")
    if manager is None:
        failures.append("EP group has no all2all manager")

    enable_taint_tracking()
    dev = torch.device("cuda")
    T, H, K = 7, 64, 2
    NT, MC = Taint("NUM_TOKS"), Taint("MODEL_CONFIG")

    def x(rows=T):
        return TaintedTensor(torch.randn(rows, H, device=dev, dtype=torch.bfloat16), (NT, MC))

    trace = os.environ.get("DOOLY_COMM_TRACE_OUT") or tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        # the engine passes its (tracer-tainted) token count
        with fc.set_forward_context(None, cfg, num_tokens=TaintedInt(T, "NUM_TOKS")):
            dpm = fc.get_forward_context().dp_metadata
            counts = None if dpm is None else [int(v) for v in dpm.num_tokens_across_dp_cpu]
            print(f"  forward context per-rank tokens: {counts}")
            if counts != [T] * DP:
                failures.append(f"per-rank token counts {counts}, want {[T] * DP}")
            else:
                with dpm.sp_local_sizes(1):
                    topk_w = torch.rand(T, K, device=dev)
                    topk_i = torch.randint(0, 8, (T, K), device=dev, dtype=torch.int32)
                    try:
                        h, w, i = ep.dispatch(x(), topk_w, topk_i)[:3]
                        shapes = [tuple(t.shape) for t in (h, w, i)]
                        print(f"  dispatch -> {shapes}")
                        if shapes != [(DP * T, H), (DP * T, K), (DP * T, K)]:
                            failures.append(f"dispatch shapes {shapes}")
                        torch.cuda.synchronize()
                        if not all(torch.equal(i[r * T:(r + 1) * T], topk_i) for r in range(DP)):
                            failures.append("gathered expert ids are not this rank's ids on every rank")
                        out = ep.combine(x(DP * T))
                        print(f"  combine  -> {tuple(out.shape)}")
                        if tuple(out.shape) != (T, H):
                            failures.append(f"combine shape {tuple(out.shape)}, want {(T, H)}")
                        if hasattr(ep, "combine_into_output"):
                            # vLLM >= 0.32 MoE runners combine into a buffer they allocate
                            buf = torch.empty(T, H, device=dev, dtype=torch.bfloat16)
                            ep.combine_into_output(x(DP * T), buf)
                            print(f"  combine_into_output -> {tuple(buf.shape)}")
                    except Exception as e:  # noqa: BLE001
                        failures.append(f"dispatch/combine raised {type(e).__name__}: {e}")
    prof.export_chrome_trace(trace)

    events = json.load(open(trace))["traceEvents"]
    nccl = [e for e in events if e.get("cat") == "kernel" and "nccl" in e.get("name", "").lower()]
    if nccl:
        failures.append(f"{len(nccl)} NCCL kernels launched: {nccl[0]['name']}")
    comm = [e for e in events if e.get("cat") == "user_annotation" and e.get("name", "").startswith("COMM: ")]
    recorded = [e["name"].split(" IN:")[0][len("COMM: "):] for e in comm]
    print(f"recorded COMM ranges: {recorded}")
    for e in comm:
        name = e["name"].split(" IN:")[0][len("COMM: "):]
        if name not in ("all_gatherv", "reduce_scatterv", "reduce_scatterv_into_output"):
            continue
        meta = parse_comm_args(e["name"].split(" ARGS:", 1)[1].strip("[]"))
        sizes = meta["args"].get("sizes")
        if meta.get("world_size") != DP or not meta.get("group", "").startswith("dp"):
            failures.append(f"{name}: group {meta.get('group')} world_size {meta.get('world_size')}")
        if not (isinstance(sizes, list) and len(sizes) == DP and all(s == {"int": T, "taint": "NUM_TOKS"} for s in sizes)):
            failures.append(f"{name}: sizes {sizes}")

    from doolyprof.profiler.parser import HierarchicalTraceParser
    parser = HierarchicalTraceParser(trace_path=trace)
    parser.parse()
    parsed = sorted(c.name for c in parser.comm_ops)
    print(f"parser comm ops: {parsed}")
    want = ["all_gatherv", "reduce_scatterv"] + (
        ["reduce_scatterv_into_output"] if hasattr(ep, "combine_into_output") else [])
    if parsed != want:
        failures.append(f"parser comm ops {parsed}, want {want}")
    # Only the caller-allocated output of reduce_scatterv_into_output is
    # receive-only (left out of the payload size); every other call sends all
    # its recorded tensors (test oracle: the collectives' definitions).
    for c in parser.comm_ops:
        ro = c.args.get("recv_only")
        expect = [1] if c.name == "reduce_scatterv_into_output" else None
        print(f"  {c.name}: recv_only={ro}")
        if ro != expect:
            failures.append(f"{c.name}: recv_only {ro}, want {expect}")

    ctx.__exit__(None, None, None)
    if failures:
        print("FAIL")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
