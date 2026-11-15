"""Real-GPU check that the comm profiler measures every recorded communication.

1. Runs comm_emulation_test.py (fake TP, one GPU) and keeps its trace: one COMM
   range per communication of the engine's group -- all_reduce, all_gather,
   all_gatherv, reduce_scatter, reduce_scatterv, gather, broadcast, send, recv.
2. Turns the ranges into collective ops exactly as trace.py does (parser +
   extract_inputs_from_taint), plans them with CommProfiler.plan, and replays
   them with CommProfiler.profile on a real TP=2 vLLM engine (2 GPUs, no fake
   TP), then saves to a scratch DB.
3. Checks that every recorded operation got a latency sweep: positive
   latencies over increasing payloads, the largest payloads (median of 5) not
   faster than the smallest (median of 5).

Usage (two free GPUs): ``python doolyprof/tests/comm_replay_test.py --gpus 0,1 [--model M] [--comm-timing graph]``
"""
import argparse
import os
import sqlite3
import statistics
import subprocess
import sys
import tempfile
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--max-size", type=int, default=8 * 1024 * 1024, help="largest payload, elements")
    ap.add_argument("--comm-timing", default="isolated", choices=["isolated", "amortized", "graph"])
    args = ap.parse_args()
    work = os.environ.get("DOOLY_REPLAY_DIR") or tempfile.mkdtemp(prefix="dooly_comm_replay_")
    os.makedirs(work, exist_ok=True)
    trace = os.path.join(work, "emulated.json")

    env = dict(os.environ, DOOLY_COMM_TRACE_OUT=trace, VLLM_FAKE_TP="2",
               CUDA_VISIBLE_DEVICES=args.gpus.split(",")[0])
    rc = subprocess.run([sys.executable, os.path.join(HERE, "comm_emulation_test.py")], env=env,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    if rc != 0 or not os.path.exists(trace):
        sys.exit(f"FAIL: comm_emulation_test.py rc={rc}")

    os.environ.pop("VLLM_FAKE_TP", None)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    import torch
    from doolyprof.profiler.parser import HierarchicalTraceParser
    from doolyprof.profiler.finput import extract_inputs_from_taint
    from doolyprof.run.comm_profiler import CommProfiler, CommProfilingConfig

    parser = HierarchicalTraceParser(trace_path=trace)
    parser.parse()
    collectives = []
    for ev in parser.comm_ops:
        inputs = extract_inputs_from_taint(taint_string=ev.params,
                                           trace_input_dims=ev.get_dims_from_taint_string(ev.params),
                                           trace_input_dtypes=[], concrete_inputs=[])
        collectives.append(SimpleNamespace(operation_name=ev.name, inputs=inputs, hierarchy=SimpleNamespace(anchor=ev)))
    recorded = sorted({c.operation_name for c in collectives})
    print(f"recorded: {recorded}")

    db = os.path.join(work, "replay.db")
    config = CommProfilingConfig(dtype=torch.bfloat16, warmup_iters=3, measure_iters=5, max_size=args.max_size,
                                 comm_timing=args.comm_timing)
    prof = CommProfiler(config=config, db_path=db)
    prof.make_db()
    prof.plan(collectives)
    prof.profile(args.model, "bfloat16", config, 2, "NCCL", "replay-test")
    prof.save_to_db(overwrite=True)

    rows = sqlite3.connect(db).execute("""
        SELECT operation, dim_param, num_dims, COUNT(*), MIN(size_bytes), MAX(size_bytes),
               MIN(latency_ms), MAX(latency_ms)
        FROM coll_op_results GROUP BY operation, dim_param, num_dims ORDER BY operation""").fetchall()
    failures = []
    measured = set()
    for op, dim, nd, n, smin, smax, lmin, lmax in rows:
        measured.add(op)
        lat = [r[0] for r in sqlite3.connect(db).execute(
            "SELECT latency_ms FROM coll_op_results WHERE operation=? ORDER BY size_bytes", (op,)).fetchall()]
        small, large = statistics.median(lat[:5]), statistics.median(lat[-5:])  # medians: single points are noisy
        print(f"  {op:16s} dim={dim} ndims={nd} rows={n:4d} bytes {smin}..{smax} latency {lmin:.4f}..{lmax:.4f} ms "
              f"(5 smallest {small:.4f}, 5 largest {large:.4f})")
        if n < 5 or lmin <= 0:
            failures.append(f"{op}: {n} rows, min latency {lmin}")
        if large < small:
            failures.append(f"{op}: largest payloads faster than smallest ({large} < {small})")
    for op in recorded:
        if op not in measured:
            failures.append(f"{op}: recorded but not measured")
    if failures:
        print("FAIL")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print(f"PASS ({work})")


if __name__ == "__main__":
    main()
