"""DoolyProf command line, the container's entry point.

    python -m doolyprof.cli run --model M --out DIR [--tp N] [--dp N] [--enable-expert-parallel]
                                [--attention-backend B] [--gpu-memory-cap F] [--max-batch-size N]
                                [--max-seq-len N] [--profile-comm] [--tracer-args=...] [--profiler-args=...]
    python -m doolyprof.cli trace   <run-tracer.py arguments>
    python -m doolyprof.cli profile <run-profiler.py arguments>
    python -m doolyprof.cli test    [suite ...]
    python -m doolyprof.cli export  --db DB --out DIR
    python -m doolyprof.cli import  DIR --db DB

``run`` traces the model (DIR/trace/, log DIR/trace.log), then profiles that
trace into DIR/profile.db (log DIR/profile.log), on the GPUs this process can
see, and exits non-zero if either stage fails. ``trace`` and ``profile`` pass
their arguments to the scripts unchanged. ``test`` runs the unit and emulation
suites in doolyprof/tests. ``export`` writes DB's measurements as a tree of CSV
files under DIR (compute/<gpu>/<engine>/<backend>/<model>/<deployment>/ and
communication/<topology>/<engine>/, see doolyprof/profiles.py); ``import`` merges
such a tree into DB, so its measurements are reused by later runs on the same GPU
and engine.
"""
import argparse
import glob
import os
import re
import shlex
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
TRACER = os.path.join(ROOT, "run", "run-tracer.py")
PROFILER = os.path.join(ROOT, "run", "run-profiler.py")
SUITES = ["tracer_ops", "taint_parse", "signature_fields", "op_args", "triton_scope", "taint_config_cache",
          "moe_routing", "compile_wrapper", "subprocess_result", "dp_ep_emulation", "sp_moe_emulation",
          "comm_emulation", "profiles"]


def _script(path, args, log=None):
    cmd = [sys.executable, path] + list(args)
    print(f"[dooly] {' '.join(shlex.quote(c) for c in cmd)}" + (f"  > {log}" if log else ""), flush=True)
    if log is None:
        return subprocess.call(cmd)
    with open(log, "w") as f:
        return subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)


def _visible_gpus():
    """The GPUs this process sees, as the scripts' --gpu value: CUDA_VISIBLE_DEVICES when it is
    set (docker/dooly-docker.sh sets it to the order its GPUs were chosen in), else '0', '0,1', ..."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return visible
    try:
        import torch
        n = torch.cuda.device_count()
    except Exception:  # noqa: BLE001
        n = 0
    return ",".join(str(i) for i in range(max(n, 1)))


def _comm_rows(db):
    """'<rows> on <topology> x<group size>, ...' from the run's coll_op_results."""
    import sqlite3
    try:
        con = sqlite3.connect(db)
        got = con.execute("SELECT topology, tp_size, COUNT(*) FROM coll_op_results "
                          "GROUP BY topology, tp_size ORDER BY tp_size").fetchall()
        con.close()
    except sqlite3.Error:
        return ""
    return ", ".join(f"{n} on {topo} x{size}" for topo, size, n in got)


def cmd_run(argv):
    p = argparse.ArgumentParser(prog="dooly run", description="Trace a model, then profile the trace.")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True, help="output folder (trace/, trace.log, profile.db, profile.log)")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--dp", type=int, default=1)
    p.add_argument("--enable-expert-parallel", action="store_true")
    p.add_argument("--attention-backend", default="FLASH_ATTN")
    p.add_argument("--gpu-memory-cap", type=float, default=None,
                   help="for both stages; raise it when the KV cache cannot hold the trace batch "
                        "(Kimi-K3 on one A100: 0.975)")
    p.add_argument("--max-batch-size", type=int, default=2)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--tracer-args", default="", help="extra run-tracer.py arguments, one string")
    p.add_argument("--profiler-args", default="",
                   help="extra run-profiler.py arguments, one string (--profiler-args='--x 1': with '=' when it "
                        "starts with '-')")
    p.add_argument("--profile-comm", action="store_true",
                   help="also measure the trace's collectives on real groups of the GPUs this run sees "
                        "(run-profiler.py --profile-comm; groups larger than that are skipped)")
    a = p.parse_args(argv)

    out = os.path.abspath(a.out)
    os.makedirs(os.path.join(out, "trace"), exist_ok=True)
    gpu = _visible_gpus()  # the tracer runs on the first (TP > 1 is emulated), the profiler on all
    par = ["--tp", str(a.tp), "--dp", str(a.dp)] + (["--enable-expert-parallel"] if a.enable_expert_parallel else [])
    cap = ["--gpu-memory-cap", str(a.gpu_memory_cap)] if a.gpu_memory_cap is not None else []

    t0 = time.time()
    rc = _script(TRACER, ["--model", a.model] + par + ["--attention-backend", a.attention_backend,
                 "--trace_dir", os.path.join(out, "trace"), "--gpu", gpu.split(",")[0]] + cap
                 + shlex.split(a.tracer_args),
                 log=os.path.join(out, "trace.log"))
    traces = sorted(glob.glob(os.path.join(out, "trace", "*.json")))
    if rc != 0 or not traces:
        print(f"[dooly] trace failed (exit {rc}); see {out}/trace.log", flush=True)
        return rc or 1
    print(f"[dooly] traced in {time.time() - t0:.0f}s: {traces[0]}", flush=True)

    t0 = time.time()
    log = os.path.join(out, "profile.log")
    rc = _script(PROFILER, ["--models", a.model, "--traces", traces[0]] + par + [
                 "--profile-output", os.path.join(out, "profile"), "--dtype", a.dtype,
                 "--attention-backend", a.attention_backend, "--max-batch-size", str(a.max_batch_size),
                 "--max-seq-len", str(a.max_seq_len), "--gpu", gpu,
                 "--db-path", os.path.join(out, "profile.db")] + cap + (["--profile-comm"] if a.profile_comm else [])
                 + shlex.split(a.profiler_args), log=log)
    text = open(log, errors="replace").read()
    rows = re.findall(r"Wrote (\d+) results", text)
    unresolved = text.count("no working parent")
    comm = ""
    if a.profile_comm:
        skipped = re.findall(r"Calls recorded on (\d+)-rank groups are not profiled", text)
        comm = ", collective measurements: " + (_comm_rows(os.path.join(out, "profile.db")) or "none")
        comm += f" (skipped {'/'.join(skipped)}-rank groups)" if skipped else ""
    print(f"[dooly] profile exit {rc} in {time.time() - t0:.0f}s: "
          f"{rows[-1] if rows else 0} result rows, {unresolved} ops without a working parent{comm}", flush=True)
    return rc or (1 if unresolved else 0)


def cmd_test(argv):
    suites = argv or SUITES
    tests = os.path.join(ROOT, "tests")
    failed = []
    for name in suites:
        path = os.path.join(tests, f"{name}_test.py")
        out = subprocess.run([sys.executable, path], capture_output=True, text=True)
        text = out.stdout + out.stderr
        passed = len(re.findall(r"^PASS", text, re.M))
        bad = re.findall(r"^(?:FAIL|ERROR) .*", text, re.M)
        print(f"{name:20s} exit {out.returncode}  pass {passed}  fail {len(bad)}", flush=True)
        for line in bad[:3]:
            print(f"    {line[:200]}")
        if out.returncode or bad:
            failed.append(name)
    print(f"[dooly] {len(suites) - len(failed)}/{len(suites)} suites passed" + (f"; failed: {failed}" if failed else ""))
    return 1 if failed else 0


def cmd_export(argv):
    p = argparse.ArgumentParser(prog="dooly export", description="Write a profile database's measurements as a tree "
                                "of CSV files (doolyprof/profiles.py). Only reads the database.")
    p.add_argument("--db", required=True, help="profile database (profile.db)")
    p.add_argument("--out", required=True, help="folder to write compute/ and communication/ into")
    a = p.parse_args(argv)
    from doolyprof.profiles import export_db
    d = export_db(a.db, a.out)
    print(f"[dooly] exported {d['deployments']} deployment(s): {d['operation_files']} operation files "
          f"({d['result_rows']} rows), {d['communication_files']} collective files ({d['communication_rows']} rows) "
          f"to {a.out}", flush=True)
    if d["skipped_deployments"] or d["skipped_communication_rows"]:
        print(f"[dooly] not exported: {d['skipped_deployments']} deployment(s) and {d['skipped_communication_rows']} "
              f"collective rows measured before GPUs and engines were recorded", flush=True)
    return 0


def cmd_import(argv):
    p = argparse.ArgumentParser(prog="dooly import", description="Merge a tree written by `dooly export` into a "
                                "profile database; rows it already has are kept.")
    p.add_argument("tree", help="folder holding compute/ and communication/")
    p.add_argument("--db", required=True, help="profile database (created if missing)")
    a = p.parse_args(argv)
    from doolyprof.profiles import import_tree
    d = import_tree(a.tree, a.db)
    print(f"[dooly] imported {d['deployments']} deployment(s) ({d['deployments_added']} new): "
          f"{d['result_rows_added']} result rows added, {d['result_rows_present']} already present; "
          f"{d['communication_rows_added']} collective rows added, {d['communication_rows_present']} already present",
          flush=True)
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    command, rest = argv[0], argv[1:]
    if command == "run":
        return cmd_run(rest)
    if command == "trace":
        return _script(TRACER, rest)
    if command == "profile":
        return _script(PROFILER, rest)
    if command == "test":
        return cmd_test(rest)
    if command == "export":
        return cmd_export(rest)
    if command == "import":
        return cmd_import(rest)
    print(f"unknown command {command!r}\n{__doc__}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
