"""How much GPU memory to hand a vLLM engine, and which GPUs nobody else is on.

vLLM refuses to start unless, at that moment, free memory on every GPU it uses is
at least total x gpu_memory_utilization. With a fixed fraction (DoolyProf used
0.9) any other process holding more than (1 - fraction) of a GPU -- even an idle
one -- stops the run, although the engine needs only the weights and a small KV
cache (DoolyProf runs with max_model_len 128 by default).

The rest of the GPU is not idle either: DoolyProf allocates its own tensors
(resolver dummy runs, profiling inputs) outside the engine's budget, and the
fixed fraction always left (1 - fraction) of the GPU for them. So
``engine_memory_fraction`` keeps that headroom and gives the engine what is free
beyond it: min(cap, free/total - (1 - cap)). On an empty GPU this is exactly the
old fraction; memory held by others comes out of the engine's (KV cache) share,
never out of DoolyProf's working memory.

Memory is read through NVML, so no CUDA context is created in the caller (which
matters for processes that spawn vLLM workers afterwards). NVML also lists the
processes on each GPU, which tells memory this run holds (an earlier stage's
tensors, in this process or in the one that started it) from other jobs' memory;
``free_gpus`` uses the same list to find the GPUs nobody else is on, by the rule
docker/dooly-docker.sh applies.
"""
import os
import time

MIN_FRACTION = 0.05  # below this the engine cannot hold a model's weights in any useful case
LOCK = "/tmp/dooly_gpu{}.lock"  # GPU i is taken by a dooly run (the lock dir docker/dooly-docker.sh uses)
BUSY_BYTES = 4 << 30  # a GPU with this much in use is busy (docker/dooly-docker.sh's threshold)
BUSY_UTIL = 5         # a GPU this busy (%) in any utilization sample is running kernels: busy (the same)
UTIL_SAMPLES, UTIL_INTERVAL = 3, 1.0  # samples a second apart: one can fall between a job's kernels
GiB = 1 << 30


def _visible_nvml_indices(count):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or visible.strip() == "":
        return list(range(count)) if visible is None else []
    out = []
    for tok in visible.split(","):
        tok = tok.strip()
        if tok.isdigit() and int(tok) < count:
            out.append(int(tok))
        else:
            return None  # UUIDs / MIG names: let the torch fallback resolve them
    return out


def _run_pids():
    """This process with its ancestors and descendants: the processes of this run. A stage runs
    in a child of the process that ran the stage before it, whose tensors are still on the GPU."""
    pids = {os.getpid()}
    try:
        import psutil
        me = psutil.Process()
        pids |= {p.pid for p in me.parents()} | {p.pid for p in me.children(recursive=True)}
    except Exception:  # noqa: BLE001 -- no psutil: this process only
        pass
    return pids


def _uid(pid):
    """The user id a process runs as, or None when it is not visible here (a container is shown
    the host's PIDs, which it cannot look up)."""
    try:
        import psutil
        return psutil.Process(pid).uids().real
    except Exception:  # noqa: BLE001
        return None


def _procs(pynvml, handle):
    """[(pid, bytes or None)] of the compute processes on a GPU; None when NVML cannot list them."""
    try:
        return [(p.pid, p.usedGpuMemory) for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)]
    except Exception:  # noqa: BLE001
        return None


def _others(procs, run, used):
    """Bytes the processes outside ``run`` hold on a GPU with ``used`` bytes in use, or None when
    NVML's list cannot tell: no list, a process without a memory figure or not visible here, or
    memory no listed process accounts for (a container is usually shown no processes)."""
    if procs is None or any(b is None for _, b in procs):
        return None
    if any(pid not in run and _uid(pid) is None for pid, _ in procs):
        return None
    if used - sum(b for _, b in procs) > GiB:
        return None
    return sum(b for pid, b in procs if pid not in run)


def _stats(devices=None):
    """[(free, total, others)] for the visible GPUs (all of them, or the first ``devices``);
    others as ``_others`` gives it (None without NVML)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            idx = _visible_nvml_indices(pynvml.nvmlDeviceGetCount())
            if idx is not None:
                idx = idx[:devices] if devices else idx
                run = _run_pids()
                out = []
                for i in idx:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    m = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    free, total = int(m.free), int(m.total)
                    out.append((free, total, _others(_procs(pynvml, handle), run, total - free)))
                return out
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001 -- no NVML: fall back to CUDA (creates a context)
        pass
    import torch
    n = torch.cuda.device_count()
    n = min(n, devices) if devices else n
    return [(*torch.cuda.mem_get_info(i), None) for i in range(n)]


def free_and_total(devices=None):
    """[(free_bytes, total_bytes)] for the visible GPUs (all of them, or the first `devices`)."""
    return [(free, total) for free, total, _ in _stats(devices)]


def _held(i, free, total, others):
    text = f"cuda:{i} {(total - free) / GiB:.1f} GiB"
    return text if others is None else f"{text}, {others / GiB:.1f} GiB of it by other processes"


def engine_memory_fraction(cap=0.9, devices=None, who="vLLM engine"):
    """gpu_memory_utilization for an engine about to start on the visible GPUs:
    min(cap, free/total - (1 - cap)) over those GPUs (see module docstring)."""
    stats = _stats(devices)
    if not stats:
        return cap
    frac = min(cap, min(free / total - (1.0 - cap) for free, total, _ in stats))
    held = "; ".join(_held(i, *s) for i, s in enumerate(stats))
    if frac < MIN_FRACTION:
        raise RuntimeError(f"[GPU] not enough free GPU memory for the {who}: in use {held}")
    if frac < cap - 0.01:  # more than ~1% of the GPU in use (NVML always shows a little)
        if all(others is not None and others < 0.01 * total for _, total, others in stats):
            held = "; ".join(_held(i, free, total, None) for i, (free, total, _) in enumerate(stats))
            print(f"[GPU] memory already in use by this run's earlier stages ({held}): giving the {who} "
                  f"{frac:.2f} of the GPU instead of {cap:.2f}. No other process is on these GPUs.", flush=True)
        else:
            print(f"[GPU] memory already in use ({held}): giving the {who} {frac:.2f} of the GPU instead of "
                  f"{cap:.2f} (DoolyProf keeps its own {1.0 - cap:.2f} headroom). If another job is running on "
                  f"these GPUs, measured latencies are not reliable.", flush=True)
    return round(frac, 3)


def free_gpus():
    """([free GPUs], {busy GPU: why}) in nvidia-smi's numbering, by docker/dooly-docker.sh's rule:
    a GPU is free when it has under 4 GiB in use, runs no kernels (utilization under BUSY_UTIL in
    each of UTIL_SAMPLES samples), has no compute process of another user and no lock dir LOCK. A
    process this machine does not show (a container is shown the host's PIDs), or one of another
    person on a shared account, cannot be told apart from ours: its memory and utilization count.
    A GPU whose utilization NVML cannot read counts as idle. None when NVML is not available."""
    try:
        import pynvml
        pynvml.nvmlInit()
    except Exception:  # noqa: BLE001
        return None
    try:
        me = os.getuid()
        free, busy = [], {}
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        peak = [0] * len(handles)   # highest utilization (%) seen
        for s in range(UTIL_SAMPLES):
            if s:
                time.sleep(UTIL_INTERVAL)
            for i, handle in enumerate(handles):
                try:
                    peak[i] = max(peak[i], int(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu))
                except Exception:  # noqa: BLE001 -- not supported on this GPU
                    pass
        for i, handle in enumerate(handles):
            m = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used = int(m.total) - int(m.free)
            others = sorted({pid for pid, _ in (_procs(pynvml, handle) or []) if _uid(pid) not in (me, None)})
            if os.path.exists(LOCK.format(i)):
                busy[i] = f"taken by another dooly run ({LOCK.format(i)})"
            elif others:
                busy[i] = "processes of other users (pid " + ", ".join(map(str, others)) + ")"
            elif used >= BUSY_BYTES:
                busy[i] = f"{used / GiB:.1f} GiB in use"
            elif peak[i] >= BUSY_UTIL:
                busy[i] = f"running kernels ({peak[i]}% utilization)"
            else:
                free.append(i)
        return free, busy
    finally:
        pynvml.nvmlShutdown()
