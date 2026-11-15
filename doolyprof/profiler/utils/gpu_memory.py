"""How much GPU memory to hand a vLLM engine.

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
matters for processes that spawn vLLM workers afterwards).
"""
import os

MIN_FRACTION = 0.05  # below this the engine cannot hold a model's weights in any useful case


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


def free_and_total(devices=None):
    """[(free_bytes, total_bytes)] for the visible GPUs (all of them, or the first `devices`)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            idx = _visible_nvml_indices(pynvml.nvmlDeviceGetCount())
            if idx is not None:
                idx = idx[:devices] if devices else idx
                infos = [pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i)) for i in idx]
                return [(int(m.free), int(m.total)) for m in infos]
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001 -- no NVML: fall back to CUDA (creates a context)
        pass
    import torch
    n = torch.cuda.device_count()
    n = min(n, devices) if devices else n
    return [torch.cuda.mem_get_info(i) for i in range(n)]


def engine_memory_fraction(cap=0.9, devices=None, who="vLLM engine"):
    """gpu_memory_utilization for an engine about to start on the visible GPUs:
    min(cap, free/total - (1 - cap)) over those GPUs (see module docstring)."""
    stats = free_and_total(devices)
    if not stats:
        return cap
    frac = min(cap, min(free / total - (1.0 - cap) for free, total in stats))
    held = ", ".join(f"cuda:{i} {(t - f) / (1 << 30):.1f} GiB" for i, (f, t) in enumerate(stats))
    if frac < MIN_FRACTION:
        raise RuntimeError(f"[GPU] not enough free GPU memory for the {who}: other processes hold {held}")
    if frac < cap - 0.01:  # more than ~1% of the GPU held elsewhere (NVML always shows a little)
        print(f"[GPU] memory already in use ({held}): giving the {who} {frac:.2f} of the GPU instead of {cap:.2f} "
              f"(DoolyProf keeps its own {1.0 - cap:.2f} headroom). If another job is running on these GPUs, "
              f"measured latencies are not reliable.", flush=True)
    return round(frac, 3)
