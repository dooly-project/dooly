"""The topology label that collective-communication measurements are stored and looked up under.

Collective latencies depend on the GPUs and on how they are linked, so the communication
profiler stores each measurement (``coll_op_results``) under a topology label, and the
compute profiler records the model's collectives (``coll_ops``) under the label of the
machine it runs on. A measurement is reused only for the same label.

``machine_topology()`` returns ``DOOLY_TOPOLOGY`` when it is set (so measurements stored
under an older label, e.g. a host name, stay reachable), and otherwise a label derived from
the hardware: the model of the first visible GPU and whether it reaches other GPUs over
NVLink, e.g. ``nvidia-a100-sxm4-80gb-nvlink`` or
``nvidia-rtx-pro-6000-blackwell-server-edition-pcie``. Machines with the same GPUs and links
share measurements. The GPU count is not part of it: a one-GPU container and a multi-GPU
communication run on the same machine must give the same label (the group size is stored
separately, as tp_size).

The hardware is read through NVML, so no CUDA context is created in the caller (which matters
for processes that spawn vLLM workers afterwards); without NVML the GPU name comes from torch
and the link type is unknown.
"""
import functools
import os
import re

NO_GPU = "no-gpu"


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")


def _nvml_name_and_nvlink():
    """(name, has an active NVLink link) of the first visible GPU, or (None, None)."""
    import pynvml
    from .gpu_memory import _visible_nvml_indices
    pynvml.nvmlInit()
    try:
        count = pynvml.nvmlDeviceGetCount()
        if count == 0:
            return None, None
        idx = _visible_nvml_indices(count)
        if idx == []:
            return None, None
        if idx is None:  # UUIDs / MIG names in CUDA_VISIBLE_DEVICES
            first = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0].strip()
            handle = pynvml.nvmlDeviceGetHandleByUUID(first)
        else:
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx[0])
        name = pynvml.nvmlDeviceGetName(handle)
        name = name.decode() if isinstance(name, bytes) else name
        nvlink = False
        for link in range(getattr(pynvml, "NVML_NVLINK_MAX_LINKS", 18)):
            try:
                if pynvml.nvmlDeviceGetNvLinkState(handle, link) == pynvml.NVML_FEATURE_ENABLED:
                    nvlink = True
                    break
            except pynvml.NVMLError:  # not supported, or no such link on this GPU
                continue
        return name, nvlink
    finally:
        pynvml.nvmlShutdown()


def topology_label(gpu_name, nvlink):
    """The label for a GPU model and link type (nvlink: True, False, or None when unknown)."""
    if not gpu_name:
        return NO_GPU
    link = {True: "nvlink", False: "pcie"}.get(nvlink, "unknown-link")
    label = _slug(gpu_name)
    return label if label.endswith("-" + link) else f"{label}-{link}"


@functools.lru_cache(maxsize=None)
def _derived_topology():
    name = nvlink = None
    try:
        name, nvlink = _nvml_name_and_nvlink()
    except Exception:  # noqa: BLE001 -- no NVML: the name from CUDA, the link unknown
        name = nvlink = None
    if name is None:
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001
            name = None
    return topology_label(name, nvlink)


def machine_topology():
    """DOOLY_TOPOLOGY if set, else the label derived from this machine's GPUs (see module docstring)."""
    return os.environ.get("DOOLY_TOPOLOGY") or _derived_topology()
