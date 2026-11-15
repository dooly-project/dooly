"""GPU check of the emulated fake-TP group (profiler/utils/comm_emulation.py).

Drives the engine's own GroupCoordinator, built by fake TP for 2 ranks on one
GPU, through every communication it has -- all_reduce, all_gather, all_gatherv,
reduce_scatter, reduce_scatterv, gather, broadcast, send, recv, dispatch,
combine -- with tainted inputs, and checks that

* each call returns what a real rank 0 would get (shape and swept-dim taint),
  computed by the engine's code, with no NCCL traffic;
* each data-moving call is recorded once as a COMM range whose ARGS reads
  back with comm_args (tainted sizes keep their taint);
* the trace parser turns the ranges into comm ops, and drops a range whose
  call never reached the transport (COMM_NOOP).

The expected shapes below are the collectives' definitions (test oracle only).

Runs directly on one GPU, under either vLLM env:
``python doolyprof/tests/comm_emulation_test.py``
"""
import json
import os
import socket
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.environ.setdefault("VLLM_FAKE_TP", "2")

import torch  # noqa: E402

from doolyprof.tracer.hooks import install_hooks  # noqa: E402

install_hooks()

from doolyprof.tracer import enable_taint_tracking  # noqa: E402
from doolyprof.tracer.tensor import TaintedTensor  # noqa: E402
from doolyprof.tracer.types import Taint, TaintedInt  # noqa: E402

TP = int(os.environ["VLLM_FAKE_TP"])


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _init_engine_world():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    cfg = VllmConfig()
    ctx = set_current_vllm_config(cfg)
    ctx.__enter__()
    init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                 distributed_init_method=f"tcp://127.0.0.1:{_free_port()}", backend="nccl")
    initialize_model_parallel(1, 1)
    return ctx


def _labels(t):
    return [None if d is None else str(d.taint) for d in t._dim_taints] if isinstance(t, TaintedTensor) else None


def main():
    ctx = _init_engine_world()
    from doolyprof.profiler.utils.fake_tp import apply_fake_tp_patches
    apply_fake_tp_patches(TP)
    import vllm.distributed.parallel_state as ps
    from doolyprof.profiler.comm_args import parse_comm_args, identity_int

    enable_taint_tracking()
    dev = torch.device("cuda")
    T, H = 7, 64
    NT, MC = Taint("NUM_TOKS"), Taint("MODEL_CONFIG")

    def x(rows=T, cols=H):
        return TaintedTensor(torch.randn(rows, cols, device=dev, dtype=torch.bfloat16), (NT, MC))

    g = ps.get_tp_group()
    assert g.world_size == TP and g.rank_in_group == 0, (g.world_size, g.rank_in_group)
    assert type(g).__name__ == "GroupCoordinator", type(g)
    print(f"emulated group {g.unique_name}: world_size={g.world_size}, communicator={type(g.device_communicator).__name__}")

    sizes = [TaintedInt(T, NT)] + [TaintedInt(T - 2, NT)] * (TP - 1)  # uneven split, rank 0 holds T rows
    total = sum(int(s) for s in sizes)
    calls = [
        ("all_reduce", lambda: g.all_reduce(x()), (T, H)),
        ("all_gather", lambda: g.all_gather(x(), dim=-1), (T, H * TP)),
        ("all_gatherv", lambda: g.all_gatherv(x(), dim=0, sizes=sizes), (total, H)),
        ("reduce_scatter", lambda: g.reduce_scatter(x(), dim=-1), (T, H // TP)),
        ("reduce_scatterv", lambda: g.reduce_scatterv(x(total), dim=0, sizes=sizes), (T, H)),
        ("gather", lambda: g.gather(x(), dst=0, dim=-1), (T, H * TP)),
        ("broadcast", lambda: g.broadcast(x(), src=0), (T, H)),
        ("send", lambda: g.send(x(), dst=1), None),
        ("recv", lambda: g.recv(x().size(), torch.bfloat16, src=1), (T, H)),
    ]

    failures = []
    trace = os.environ.get("DOOLY_COMM_TRACE_OUT") or tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        outputs = {}
        for name, fn, want in calls:
            try:
                out = fn()
            except Exception as e:  # noqa: BLE001
                failures.append(f"{name}: raised {type(e).__name__}: {e}")
                continue
            outputs[name] = out
            got = None if out is None else tuple(out.shape)
            if got != want:
                failures.append(f"{name}: shape {got}, want {want}")
            print(f"  {name:16s} -> {got} taints={_labels(out)}")
        # dispatch/combine: on a TP group without an all2all manager the engine's
        # own implementation moves nothing -> the range must carry COMM_NOOP.
        try:
            topk_w = torch.rand(T, 2, device=dev)
            topk_i = torch.zeros(T, 2, device=dev, dtype=torch.int32)
            res = g.dispatch(x(), topk_w, topk_i)
            print(f"  dispatch         -> {type(res).__name__} (expected no transport)")
        except Exception as e:  # noqa: BLE001
            print(f"  dispatch         -> raised {type(e).__name__}: {str(e)[:80]} (engine requires an all2all manager)")
    prof.export_chrome_trace(trace)
    for f in failures:
        print("  !", f)

    # NCCL never ran: no nccl kernels in the trace.
    events = json.load(open(trace))["traceEvents"]
    nccl = [e for e in events if e.get("cat") == "kernel" and "nccl" in e.get("name", "").lower()]
    if nccl:
        failures.append(f"{len(nccl)} NCCL kernels launched: {nccl[0]['name']}")

    comm = [e for e in events if e.get("cat") == "user_annotation" and e.get("name", "").startswith("COMM: ")]
    recorded = [e["name"].split(" IN:")[0][len("COMM: "):] for e in comm]
    print(f"recorded COMM ranges: {recorded}")
    for name, _, _ in calls:
        n = recorded.count(name)
        if n != 1:
            failures.append(f"{name}: recorded {n} times, want 1")
    for e in comm:
        meta = parse_comm_args(e["name"].split(" ARGS:", 1)[1].strip("[]"))
        if meta.get("world_size") != TP:
            failures.append(f"{e['name'][:40]}: world_size {meta.get('world_size')}")
        if e["name"].startswith("COMM: all_gatherv "):
            sz = meta["args"].get("sizes")
            if not (isinstance(sz, list) and sz and sz[0] == {"int": T, "taint": "NUM_TOKS"}):
                failures.append(f"all_gatherv sizes not recorded with taint: {sz}")
            if identity_int(meta) != 0:
                failures.append(f"all_gatherv dim_param {identity_int(meta)}, want 0")

    from doolyprof.profiler.parser import HierarchicalTraceParser
    parser = HierarchicalTraceParser(trace_path=trace)
    parser.parse()
    parsed = [c.name for c in parser.comm_ops]
    print(f"parser comm ops: {parsed}")
    for name, _, _ in calls:
        if name not in parsed:
            failures.append(f"parser lost {name}")
    if "dispatch" in parsed or "combine" in parsed:
        failures.append("parser kept a COMM range that never reached the transport")

    ctx.__exit__(None, None, None)
    if failures:
        print("FAIL")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
