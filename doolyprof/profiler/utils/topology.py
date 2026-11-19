"""The topology label that collective-communication measurements are stored and looked up under,
and the choice of GPUs to measure them on.

Collective latencies depend on the GPUs and on how they are linked, so the communication
profiler stores each measurement (``coll_op_results``) under a topology label, and the
compute profiler records the model's collectives (``coll_ops``) under the label for the
group size each one ran on. A measurement is reused only for the same label and group size.

``group_topology(group_size)`` returns ``DOOLY_TOPOLOGY`` when it is set (so measurements
stored under an older label, e.g. a host name, stay reachable), and otherwise a label
derived from the hardware: the model of the first visible GPU and whether a group of
``group_size`` GPUs is linked by NVLink, e.g. ``nvidia-a100-80gb-pcie-nvlink`` or
``nvidia-rtx-pro-6000-blackwell-server-edition-pcie``.

- When the process sees ``group_size`` GPUs (a communication worker), the link is that of
  the first ``group_size`` visible GPUs, the ones the group's ranks run on: NVLink only if
  every pair of them is. So a pair of GPUs that are not linked to each other is measured
  under the ``-pcie`` label even when each has NVLink links to other GPUs, and its
  latencies never stand in for NVLink ones.
- When it sees fewer (the compute profiler in a one-GPU container), the link is what the
  first GPU's NVLink links allow: NVLink if the group fits in the GPU's NVLink domain
  (itself and the GPUs at the other ends of its links, or every GPU when a link goes to an
  NVLink switch). A two-GPU group on a machine of NVLink pairs is ``-nvlink`` and a
  four-GPU group is ``-pcie``, the same labels a communication run on the best GPUs gives.

The GPU count is not part of the label: the group size is stored separately (tp_size).
``machine_topology()`` is the label without a group (NVLink if the GPU has an active link).

The hardware is read through NVML, so no CUDA context is created in the caller (which matters
for processes that spawn vLLM workers afterwards); without NVML the GPU name comes from torch
and the link type is unknown.

``pick_gpus()`` chooses the GPUs for a run from ``nvidia-smi topo -m``: the free GPUs whose
weakest pairwise link is the strongest, in an order where each group of the first k GPUs is
as well linked as it can be. docker/dooly-docker.sh runs this file as a script on the host
(``python3 -I topology.py pick N FREE``, reading the matrix on stdin), so the module imports
only the standard library at the top.
"""
import functools
import itertools
import os
import re
import sys

NO_GPU = "no-gpu"


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")


# --------------------------------------------------------------------------------------------
# Links between the GPUs a process sees (NVML)
# --------------------------------------------------------------------------------------------

def _visible_handles(pynvml):
    from .gpu_memory import _visible_nvml_indices
    count = pynvml.nvmlDeviceGetCount()
    idx = _visible_nvml_indices(count)
    if idx is None:  # UUIDs / MIG names in CUDA_VISIBLE_DEVICES
        return [pynvml.nvmlDeviceGetHandleByUUID(tok.strip())
                for tok in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if tok.strip()]
    return [pynvml.nvmlDeviceGetHandleByIndex(i) for i in idx]


def _nvlink_domain(pynvml, handle):
    """The size of the NVLink domain of ``handle``: 1 without an active link, None when a
    link reaches an NVLink switch (every GPU on it) or the other ends cannot be read, else
    1 + the number of GPUs at the other ends of its links."""
    peers, active = set(), False
    switch = getattr(pynvml, "NVML_NVLINK_DEVICE_TYPE_SWITCH", 2)
    for link in range(getattr(pynvml, "NVML_NVLINK_MAX_LINKS", 18)):
        try:
            if pynvml.nvmlDeviceGetNvLinkState(handle, link) != pynvml.NVML_FEATURE_ENABLED:
                continue
        except pynvml.NVMLError:  # not supported, or no such link on this GPU
            continue
        active = True
        try:
            if pynvml.nvmlDeviceGetNvLinkRemoteDeviceType(handle, link) == switch:
                return None
            peers.add(pynvml.nvmlDeviceGetNvLinkRemotePciInfo(handle, link).busId)
        except (pynvml.NVMLError, AttributeError):
            return None
    return 1 + len(peers) if active else 1


def _nvlinked(pynvml, a, b):
    try:
        return pynvml.nvmlDeviceGetP2PStatus(a, b, pynvml.NVML_P2P_CAPS_INDEX_NVLINK) == pynvml.NVML_P2P_STATUS_OK
    except (pynvml.NVMLError, AttributeError):
        return None


def _nvml_links(group_size):
    """(name of the first visible GPU, linked, fits) for a group of ``group_size`` GPUs, or
    (None, None, None). ``linked``: whether every pair of the first ``group_size`` visible
    GPUs is linked by NVLink (None with fewer visible GPUs or when NVML cannot tell);
    ``fits``: whether the group fits in the first GPU's NVLink domain (with no group size:
    whether the GPU has an active link)."""
    import pynvml
    pynvml.nvmlInit()
    try:
        handles = _visible_handles(pynvml)
        if not handles:
            return None, None, None
        name = pynvml.nvmlDeviceGetName(handles[0])
        name = name.decode() if isinstance(name, bytes) else name
        domain = _nvlink_domain(pynvml, handles[0])
        size = group_size or 2
        fits = domain != 1 and (domain is None or size <= domain)
        linked = None
        if group_size and group_size >= 2 and len(handles) >= group_size:
            pairs = [_nvlinked(pynvml, a, b) for a, b in itertools.combinations(handles[:group_size], 2)]
            linked = False if False in pairs else (None if None in pairs else True)
        return name, linked, fits
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
def _links(group_size=None):
    try:
        return _nvml_links(group_size)
    except Exception:  # noqa: BLE001 -- no NVML: the name from CUDA, the link unknown
        return None, None, None


@functools.lru_cache(maxsize=None)
def _derived_topology(group_size=None):
    name, linked, fits = _links(group_size)
    if name is None:
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001
            name = None
    return topology_label(name, linked if linked is not None else fits)


def group_topology(group_size):
    """DOOLY_TOPOLOGY if set, else the label for a group of ``group_size`` GPUs (see the module docstring)."""
    return os.environ.get("DOOLY_TOPOLOGY") or _derived_topology(group_size)


def machine_topology():
    """DOOLY_TOPOLOGY if set, else the label of this machine's GPUs without a group size."""
    return os.environ.get("DOOLY_TOPOLOGY") or _derived_topology(None)


def group_link_warning(group_size):
    """A warning when the first ``group_size`` visible GPUs are not all linked by NVLink
    although the first one could be in such a group, else None."""
    if os.environ.get("DOOLY_TOPOLOGY"):
        return None
    _, linked, fits = _links(group_size)
    if linked is False and fits:
        return (f"the {group_size} GPUs of this group are not all linked by NVLink, though GPU 0 has NVLink "
                f"links to others: measurements are stored under {group_topology(group_size)}. "
                f"Give the run NVLink-connected GPUs (docker/dooly-docker.sh picks them).")
    return None


# --------------------------------------------------------------------------------------------
# Choosing the GPUs for a run (nvidia-smi topo -m, on the host)
# --------------------------------------------------------------------------------------------

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
# nvidia-smi topo -m: PIX one PCIe switch, PXB several, PHB a host bridge, NODE host bridges
# of one NUMA node, SYS across NUMA nodes; NV<k> k NVLink links.
_PCIE_RANK = {"PIX": 5, "PXB": 4, "PHB": 3, "NODE": 2, "SYS": 1}
_NO_PAIR = (99, 0)


def link_rank(cls):
    """(class rank, NVLink count) of a ``nvidia-smi topo -m`` entry; higher is faster."""
    m = re.fullmatch(r"NV(\d+)", cls)
    if m:
        return (6, int(m.group(1)))
    return (_PCIE_RANK.get(cls, 0), 0)


def parse_topo_matrix(text):
    """{gpu: {gpu: class}} from ``nvidia-smi topo -m`` output: the header line is indented,
    each GPU's row starts with its name."""
    lines = [_ANSI.sub("", ln).rstrip() for ln in text.splitlines()]
    header = next((ln.split() for ln in lines if ln[:1].isspace() and ln.split()[:1] == ["GPU0"]), None)
    if header is None:
        return {}
    cols = [int(c[3:]) if re.fullmatch(r"GPU\d+", c) else None for c in header]
    links = {}
    for ln in lines:
        cells = ln.split()
        if ln[:1].isspace() or not cells or not re.fullmatch(r"GPU\d+", cells[0]):
            continue
        row = int(cells[0][3:])
        for col, cls in zip(cols, cells[1:]):
            if col is not None and col != row:
                links.setdefault(row, {})[col] = cls
    return links


def _strength(gpus, links):
    pairs = [link_rank(links.get(a, {}).get(b, "")) for a, b in itertools.combinations(gpus, 2)]
    return (min(pairs), tuple(sorted(pairs))) if pairs else (_NO_PAIR, ())


def _ordered(gpus, links):
    """``gpus`` in an order where each prefix is as well linked as the set allows: the best
    pair first, then each time the GPU with the strongest weakest link to those placed."""
    gpus = sorted(gpus)
    if len(gpus) <= 2:
        return gpus
    first = max(itertools.combinations(gpus, 2), key=lambda p: (link_rank(links.get(p[0], {}).get(p[1], "")),
                                                                 [-g for g in p]))
    order = list(first)
    rest = [g for g in gpus if g not in order]
    while rest:
        nxt = max(rest, key=lambda g: (min(link_rank(links.get(g, {}).get(o, "")) for o in order), -g))
        order.append(nxt)
        rest.remove(nxt)
    return order


def _best_set(n, gpus, links):
    best = None
    for combo in itertools.combinations(sorted(gpus), n):
        key = _strength(combo, links)
        if best is None or key > best[0]:
            best = (key, combo)
    return None if best is None else list(best[1])


def link_class(gpus, links):
    """The ``nvidia-smi topo -m`` class of the weakest link among ``gpus`` ("-" for one GPU)."""
    weakest = None
    for a, b in itertools.combinations(gpus, 2):
        cls = links.get(a, {}).get(b, "?")
        if weakest is None or link_rank(cls) < link_rank(weakest):
            weakest = cls
    return weakest or "-"


def pick_gpus(n, free, links):
    """(chosen GPUs in order, class of their weakest link, class of the weakest link of the
    machine's best n GPUs), or None when fewer than n GPUs are free."""
    free = sorted(set(free))
    if len(free) < n:
        return None
    every = sorted(set(links) | set(free))
    chosen = _ordered(_best_set(n, free, links), links)
    best = _best_set(n, every, links)
    return chosen, link_class(chosen, links), link_class(best, links)


def link_report(gpus, links):
    """(class of the weakest link among ``gpus``, class of the weakest link of the machine's best
    len(gpus) GPUs)."""
    every = sorted(set(links) | set(gpus))
    return link_class(gpus, links), link_class(_best_set(len(gpus), every, links), links)


def same_class(a, b):
    return link_rank(a)[0] == link_rank(b)[0] if a != "-" and b != "-" else a == b


def _main(argv):
    """The ``nvidia-smi topo -m`` matrix on stdin, GPU lists comma-separated.
    pick N FREE: prints ``CHOSEN LINK BEST SAME`` (SAME 1 when the chosen GPUs are linked like
    the machine's best N); exit status 1 when fewer than N GPUs are free.
    link GPUS: prints ``LINK BEST SAME`` for the GPUs named."""
    if len(argv) == 2 and argv[0] == "link":
        gpus = [int(g) for g in argv[1].split(",") if g.strip()]
        link, best = link_report(gpus, parse_topo_matrix(sys.stdin.read()))
        print(link, best, int(same_class(link, best)))
        return 0
    if len(argv) != 3 or argv[0] != "pick":
        print(_main.__doc__, file=sys.stderr)
        return 2
    n = int(argv[1])
    free = [int(g) for g in argv[2].split(",") if g.strip()]
    got = pick_gpus(n, free, parse_topo_matrix(sys.stdin.read()))
    if got is None:
        return 1
    chosen, link, best = got
    print(",".join(map(str, chosen)), link, best, int(same_class(link, best)))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
