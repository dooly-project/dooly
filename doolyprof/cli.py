"""DoolyProf command line, the container's entry point.

    python -m doolyprof.cli run --model M --out DIR [--db DB] [--tp N] [--dp N] [--enable-expert-parallel]
                                [--attention-backend B] [--gpu-memory-cap F] [--max-batch-size N]
                                [--max-seq-len N] [--profile-comm] [--gpus N]
                                [--tracer-args=...] [--profiler-args=...]
    python -m doolyprof.cli trace   <run-tracer.py arguments>
    python -m doolyprof.cli profile <run-profiler.py arguments>
    python -m doolyprof.cli test    [suite ...]
    python -m doolyprof.cli export  --db DB --out DIR
    python -m doolyprof.cli import  DIR --db DB
    python -m doolyprof.cli verify  DIR

``run`` traces the model (DIR/trace/, log DIR/trace.log), then profiles that
trace into DIR/profile.db, or DB (log DIR/profile.log), and exits non-zero if
either stage fails. It runs on the GPUs CUDA_VISIBLE_DEVICES names when that is set
(docker/dooly-docker.sh sets it), else on GPUs nobody else is on: one, or with
--profile-comm one per rank of the largest group (--gpus N to say how many).
Results DB already has for this
GPU and engine (e.g. imported from DoolyDB) are reused, not measured again. ``trace`` and ``profile`` pass
their arguments to the scripts unchanged. ``test`` runs the unit and emulation
suites in doolyprof/tests. ``export`` writes DB's measurements as a tree of CSV
files under DIR (compute/<gpu>/<engine>/<backend>/<model>/<deployment>/ and
communication/<topology>/<engine>/, see doolyprof/profiles.py); ``import`` merges
such a tree into DB, so its measurements are reused by later runs on the same GPU
and engine; ``verify`` checks that a tree is exactly what ``export`` writes.
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
          "comm_emulation", "profiles", "cli_run"]


def _script(path, args, log=None):
    cmd = [sys.executable, path] + list(args)
    print(f"[dooly] {' '.join(shlex.quote(c) for c in cmd)}" + (f"  > {log}" if log else ""), flush=True)
    if log is None:
        return subprocess.call(cmd)
    with open(log, "w") as f:
        return subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)


def _release(locks):
    for lock in locks:
        try:
            os.rmdir(lock)
        except OSError:
            pass


def _choose_gpus(n):
    """(--gpu value, lock dirs claimed, None), or (None, [], why not). The GPUs CUDA_VISIBLE_DEVICES
    names when it is set (docker/dooly-docker.sh sets it to the GPUs it chose; set it yourself to pick
    GPUs); else n GPUs nobody else is on (gpu_memory.free_gpus: docker/dooly-docker.sh's rule), as
    well linked as the free ones allow, each claimed for the run with docker/dooly-docker.sh's lock
    dir so that two runs never share a GPU."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return visible, [], None
    from doolyprof.profiler.utils.gpu_memory import LOCK, free_gpus
    from doolyprof.profiler.utils.topology import link_class, parse_topo_matrix, pick_gpus
    got = free_gpus()
    if got is None:
        gpus = ",".join(str(i) for i in range(n))
        print(f"[dooly] cannot tell which GPUs are free (no NVML): using GPU(s) {gpus}; set CUDA_VISIBLE_DEVICES "
              f"to choose", flush=True)
        return gpus, [], None
    free, busy = got
    try:
        topo = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=60).stdout
    except Exception:  # noqa: BLE001 -- no nvidia-smi: links unknown, the first free GPUs
        topo = ""
    links = parse_topo_matrix(topo)
    picked = pick_gpus(n, free, links)
    if picked is None:
        why = "; ".join(f"GPU {i}: {reason}" for i, reason in sorted(busy.items()))
        return None, [], (f"{n} free GPU(s) needed, {len(free)} free ({why}). Latencies measured next to another "
                          f"job are not reliable; to use particular GPUs anyway, set CUDA_VISIBLE_DEVICES.")
    chosen, claimed = picked[0], []
    for g in chosen:
        try:
            os.mkdir(LOCK.format(g))
        except OSError:
            _release(claimed)
            return None, [], f"GPU {g} was just taken by another dooly run ({LOCK.format(g)}); run again"
        claimed.append(LOCK.format(g))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # the scripts' CUDA_VISIBLE_DEVICES in nvidia-smi's numbering
    gpus = ",".join(map(str, chosen))
    print(f"[dooly] GPU(s) {gpus}" + (f", linked by {link_class(chosen, links)}" if n > 1 and links else "")
          + (f" (in use elsewhere: {', '.join(map(str, sorted(busy)))})" if busy else ""), flush=True)
    return gpus, claimed, None


def _nvcc_note():
    """A note when the nvcc kernels would be compiled with is missing or of another CUDA major version
    than PyTorch's build: FlashInfer compiles some kernels the first time they are used, with the
    nvcc of CUDA_HOME (else CUDA_PATH, the nvcc on PATH, /usr/local/cuda: PyTorch's CUDA_HOME
    finds the same). None when they match or PyTorch has no CUDA."""
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME
        want = torch.version.cuda
    except Exception:  # noqa: BLE001
        return None
    if not want:
        return None
    nvcc, have = os.path.join(CUDA_HOME, "bin", "nvcc") if CUDA_HOME else None, None
    if nvcc and os.path.exists(nvcc):
        try:
            out = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=60).stdout
            m = re.search(r"release (\d+)\.(\d+)", out)
            have = f"{m.group(1)}.{m.group(2)}" if m else None
        except Exception:  # noqa: BLE001
            pass
    major = want.split(".")[0]
    if have and have.split(".")[0] == major:
        return None
    found = f"{nvcc} is CUDA {have}" if have else "no nvcc was found"
    return (f"[dooly] note: FlashInfer compiles some kernels the first time they are used, with the nvcc of "
            f"CUDA_HOME (else the nvcc on PATH): {found}, and PyTorch is built for CUDA {want}. If the trace or "
            f"profile stops with an nvcc error, set CUDA_HOME to a CUDA {major} toolkit and put its bin/ first on "
            f"PATH.")


def _last_run(db):
    """The highest run_id in DB's profile_runs: 0 for a new database (the rows a run writes carry
    a higher one)."""
    import sqlite3
    if not os.path.exists(db):
        return 0
    try:
        con = sqlite3.connect(db)
        try:
            return con.execute("SELECT COALESCE(MAX(run_id), 0) FROM profile_runs").fetchone()[0]
        finally:
            con.close()
    except sqlite3.Error:
        return 0


def _run_rows(db, after):
    """(result rows, collective rows by topology and group size) the runs after run_id ``after``
    wrote into DB; (None, []) when DB cannot tell."""
    import sqlite3
    try:
        con = sqlite3.connect(db)
        try:
            rows = con.execute("SELECT COUNT(*) FROM results WHERE run_id > ?", (after,)).fetchone()[0]
            try:
                comm = con.execute("SELECT topology, tp_size, COUNT(*) FROM coll_op_results WHERE run_id > ? "
                                   "GROUP BY topology, tp_size ORDER BY tp_size", (after,)).fetchall()
            except sqlite3.Error:  # no collective was measured into this database yet
                comm = []
            return rows, comm
        finally:
            con.close()
    except sqlite3.Error:
        return None, []


def cmd_run(argv):
    p = argparse.ArgumentParser(prog="dooly run", description="Trace a model, then profile the trace.")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True, help="output folder (trace/, trace.log, profile.db, profile.log)")
    p.add_argument("--db", default=None,
                   help="profile database to write and reuse results from (default: OUT/profile.db), e.g. one "
                        "DoolyDB was imported into")
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
                   help="also measure the trace's collectives on real groups of the GPUs this run gets "
                        "(run-profiler.py --profile-comm; groups larger than that are skipped)")
    p.add_argument("--gpus", type=int, default=None,
                   help="how many free GPUs to take when CUDA_VISIBLE_DEVICES is not set (default 1; with "
                        "--profile-comm one per rank of the largest group: tp, or tp x dp with expert parallelism)")
    a = p.parse_args(argv)

    n = a.gpus or ((a.tp * a.dp if a.enable_expert_parallel else a.tp) if a.profile_comm else 1)
    gpu, locks, why = _choose_gpus(n)
    if gpu is None:
        print(f"[dooly] not run: {why}", flush=True)
        return 1
    try:
        return _run(a, gpu)
    finally:
        _release(locks)


def _run(a, gpu):
    out = os.path.abspath(a.out)
    os.makedirs(os.path.join(out, "trace"), exist_ok=True)
    note = _nvcc_note()
    if note:
        print(note, flush=True)
    # the tracer runs on the first GPU (TP > 1 is emulated), the profiler measures computation on the
    # first and collectives on groups of all of them
    par = ["--tp", str(a.tp), "--dp", str(a.dp)] + (["--enable-expert-parallel"] if a.enable_expert_parallel else [])
    cap = ["--gpu-memory-cap", str(a.gpu_memory_cap)] if a.gpu_memory_cap is not None else []

    t0 = time.time()
    rc = _script(TRACER, ["--model", a.model] + par + ["--attention-backend", a.attention_backend,
                 "--trace_dir", os.path.join(out, "trace"), "--gpu", gpu.split(",")[0]] + cap
                 + shlex.split(a.tracer_args),
                 log=os.path.join(out, "trace.log"))
    # the trace this run just wrote: an --out folder used before also holds earlier models' traces
    traces = sorted((t for t in glob.glob(os.path.join(out, "trace", "*.json")) if os.path.getmtime(t) >= t0 - 1),
                    key=os.path.getmtime)
    if rc != 0 or not traces:
        print(f"[dooly] trace failed (exit {rc}); see {out}/trace.log", flush=True)
        return rc or 1
    trace = traces[-1]
    print(f"[dooly] traced in {time.time() - t0:.0f}s: {trace}", flush=True)
    db = os.path.abspath(a.db) if a.db else os.path.join(out, "profile.db")

    t0 = time.time()
    log = os.path.join(out, "profile.log")
    before = _last_run(db)
    rc = _script(PROFILER, ["--models", a.model, "--traces", trace] + par + [
                 "--profile-output", os.path.join(out, "profile"), "--dtype", a.dtype,
                 "--attention-backend", a.attention_backend, "--max-batch-size", str(a.max_batch_size),
                 "--max-seq-len", str(a.max_seq_len), "--gpu", gpu,
                 "--db-path", db] + cap + (["--profile-comm"] if a.profile_comm else [])
                 + shlex.split(a.profiler_args), log=log)
    text = open(log, errors="replace").read()
    # the rows this run added to the database (a batch shape measured twice in one run is one row)
    rows, coll = _run_rows(db, before)
    if rows is None:
        rows = sum(int(n) for n in re.findall(r"Wrote (\d+) results", text))
    unresolved = text.count("no working parent")
    comm = ""
    if a.profile_comm:
        skipped = re.findall(r"Calls recorded on (\d+)-rank groups are not profiled", text)
        comm = ", collective measurements: " + (", ".join(f"{k} on {topo} x{size}" for topo, size, k in coll)
                                                 or "none")
        comm += f" (skipped {'/'.join(skipped)}-rank groups)" if skipped else ""
    print(f"[dooly] profile exit {rc} in {time.time() - t0:.0f}s: "
          f"{rows} result rows, {unresolved} ops without a working parent{comm}", flush=True)
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
    if d["duplicate_rows"]:
        print(f"[dooly] not exported: {d['duplicate_rows']} second measurement(s) of an operation at a batch shape "
              f"(e.g. by runs of different TP degrees); the first is kept", flush=True)
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


def cmd_verify(argv):
    p = argparse.ArgumentParser(prog="dooly verify", description="Check that a tree (dooly export's layout, e.g. a "
                                "DoolyDB contribution) imports and is exactly what dooly export writes.")
    p.add_argument("tree", help="folder holding compute/ and communication/")
    a = p.parse_args(argv)
    from doolyprof.profiles import verify_tree
    problems, counts = verify_tree(a.tree)
    for line in problems[:50]:
        print(f"[dooly] {line}", flush=True)
    if problems:
        print(f"[dooly] {len(problems)} problem(s) in {a.tree}", flush=True)
        return 1
    print(f"[dooly] {a.tree} is valid: {counts.get('deployments', 0)} deployment(s), "
          f"{counts.get('result_rows_added', 0)} result rows, {counts.get('communication_rows_added', 0)} "
          f"collective rows", flush=True)
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
    if command == "verify":
        return cmd_verify(rest)
    print(f"unknown command {command!r}\n{__doc__}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
