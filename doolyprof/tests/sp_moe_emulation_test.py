"""GPU check of fake TP x DP with expert parallelism: sequence-parallel MoE.

With TP > 1, DP > 1 and EP, vLLM's MoE runs sequence parallel
(``ParallelConfig.use_sequence_parallel_moe``): every TP rank takes
ceil(T / TP) of its DP rank's T tokens, and dispatch / combine move them over
the TP x DP expert-parallel group. Fake TP2 x DP2 on one GPU; checks that

* the forward context, set with the deployment's parallel configuration
  (fake_tp.fake_parallel_vllm_config), gives per-rank sizes
  [ceil(T / TP)] * (TP * DP), each keeping the NUM_TOKS taint;
* dispatch over the 4-rank EP group returns 4 * ceil(T / TP) rows and combine
  this rank's ceil(T / TP) rows;
* the trace records the transfers on the EP group (world size 4) with tainted
  per-rank sizes, so the comm profiler replays them on a 4-rank group.

Runs directly on one GPU, under any vLLM env:
``python doolyprof/tests/sp_moe_emulation_test.py``
"""
import inspect
import json
import os
import socket
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.environ["VLLM_FAKE_TP"] = "2"
os.environ["VLLM_FAKE_DP"] = "2"
os.environ["VLLM_FAKE_EP"] = "1"

import torch  # noqa: E402

from doolyprof.tracer.hooks import install_hooks  # noqa: E402

install_hooks()

from doolyprof.tracer import enable_taint_tracking  # noqa: E402
from doolyprof.tracer.tensor import TaintedTensor  # noqa: E402
from doolyprof.tracer.types import Taint, TaintedInt  # noqa: E402

TP, DP = 2, 2


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
    apply_fake_tp_patches(TP, DP)
    import vllm.distributed.parallel_state as ps
    import vllm.forward_context as fc
    from doolyprof.profiler.comm_args import parse_comm_args

    failures = []
    ep = ps.get_ep_group()
    if ep.world_size != TP * DP:
        failures.append(f"ep group size {ep.world_size}, want {TP * DP}")

    enable_taint_tracking()
    dev = torch.device("cuda")
    T, H, K = 7, 64, 2
    C = -(-T // TP)  # tokens per sequence-parallel rank
    NT, MC = Taint("NUM_TOKS"), Taint("MODEL_CONFIG")

    def x(rows):
        return TaintedTensor(torch.randn(rows, H, device=dev, dtype=torch.bfloat16), (NT, MC))

    trace = os.environ.get("DOOLY_COMM_TRACE_OUT") or tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with fc.set_forward_context(None, cfg, num_tokens=TaintedInt(T, "NUM_TOKS")):
            dpm = fc.get_forward_context().dp_metadata
            if dpm is None:
                failures.append("no DP metadata in the forward context")
            else:
                with dpm.sp_local_sizes(TP):
                    sizes = dpm.get_chunk_sizes_across_dp_rank()
                    print(f"  sequence-parallel per-rank sizes: {[repr(s) for s in sizes]}")
                    if [int(s) for s in sizes] != [C] * (TP * DP):
                        failures.append(f"per-rank sizes {sizes}, want {[C] * (TP * DP)}")
                    if not all(getattr(s, "taint", None) == NT for s in sizes):
                        failures.append(f"per-rank sizes lost the NUM_TOKS taint: {[repr(s) for s in sizes]}")
                    params = inspect.signature(ep.dispatch).parameters
                    try:
                        if "topk_ids" in params:
                            args = (x(C), torch.rand(C, K, device=dev),
                                    torch.randint(0, 8, (C, K), device=dev, dtype=torch.int32))
                        else:  # vLLM 0.17.1: dispatch(hidden_states, router_logits)
                            args = (x(C), torch.rand(C, 8, device=dev, dtype=torch.bfloat16))
                        out = ep.dispatch(*args, is_sequence_parallel=True)
                        rows = [tuple(t.shape)[0] for t in out[:len(args)]]
                        print(f"  dispatch -> rows {rows}")
                        if rows != [TP * DP * C] * len(args):
                            failures.append(f"dispatch rows {rows}, want {TP * DP * C}")
                        comb = ep.combine(x(TP * DP * C), is_sequence_parallel=True)
                        print(f"  combine  -> {tuple(comb.shape)}")
                        if tuple(comb.shape) != (C, H):
                            failures.append(f"combine shape {tuple(comb.shape)}, want {(C, H)}")
                    except Exception as e:  # noqa: BLE001
                        failures.append(f"dispatch/combine raised {type(e).__name__}: {e}")
    prof.export_chrome_trace(trace)

    events = json.load(open(trace))["traceEvents"]
    comm = [e for e in events if e.get("cat") == "user_annotation" and e.get("name", "").startswith("COMM: ")]
    for e in comm:
        name = e["name"].split(" IN:")[0][len("COMM: "):]
        meta = parse_comm_args(e["name"].split(" ARGS:", 1)[1].strip("[]"))
        print(f"  recorded {name} on {meta.get('group')} ({meta.get('world_size')})")
        if name in ("all_gatherv", "reduce_scatterv"):
            sizes = meta["args"].get("sizes")
            if meta.get("world_size") != TP * DP:
                failures.append(f"{name}: world_size {meta.get('world_size')}, want {TP * DP}")
            if sizes != [{"int": C, "taint": "NUM_TOKS"}] * (TP * DP):
                failures.append(f"{name}: sizes {sizes}")
    from doolyprof.profiler.parser import HierarchicalTraceParser
    parser = HierarchicalTraceParser(trace_path=trace)
    parser.parse()
    parsed = sorted(c.name for c in parser.comm_ops)
    print(f"parser comm ops: {parsed}")
    if parsed != ["all_gatherv", "reduce_scatterv"]:
        failures.append(f"parser comm ops {parsed}, want ['all_gatherv', 'reduce_scatterv']")

    ctx.__exit__(None, None, None)
    if failures:
        print("FAIL")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
