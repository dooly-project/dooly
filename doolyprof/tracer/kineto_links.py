"""Repair Kineto kernel -> CPU-event links in an exported chrome trace.

Kineto attributes a GPU kernel to the CPU op that launched it through the
``External id`` it copies from the enclosing op onto the kernel. Kernels
launched through the CUDA *driver* API outside any aten op -- Triton's
``cuLaunchKernelEx``, used by vLLM 0.32's KDA / GDN / gated layer-norm paths --
come out with only a ``correlation`` id, so the trace parser cannot attach them
to anything and they vanish from the profile.

The correlation id still names the launch event (``cuda_runtime`` /
``cuda_driver``), whose timestamp lies inside the record_function scopes active
at launch (the tracer's MODULE / API annotations, or a cpu op). This module
gives each such kernel the ``External id`` of the innermost CPU event enclosing
its launch on the launching thread: the same link Kineto makes for runtime
launches, rebuilt from Kineto's own correlation data. Run it before the launch
events are dropped from the exported trace.
"""
import bisect
from collections import defaultdict

LAUNCH_CATEGORIES = ("cuda_runtime", "cuda_driver")
ENCLOSING_CATEGORIES = ("cpu_op", "user_annotation")


def attach_launch_external_ids(events) -> int:
    """Set ``args['External id']`` on kernels that lack it, from the innermost
    CPU event enclosing their launch. Mutates ``events``; returns the number
    of kernels linked."""
    launches = {}
    for e in events:
        if e.get("cat") in LAUNCH_CATEGORIES and e.get("ph") == "X":
            corr = (e.get("args") or {}).get("correlation")
            if corr is not None:
                launches[corr] = e

    # Enclosing candidates per launching thread, sorted by start time.
    by_thread = defaultdict(list)
    for e in events:
        if (e.get("cat") in ENCLOSING_CATEGORIES and e.get("ph") == "X"
                and "External id" in (e.get("args") or {})):
            by_thread[(e.get("pid"), e.get("tid"))].append(e)
    starts = {}
    for key, evs in by_thread.items():
        evs.sort(key=lambda x: (x["ts"], -x.get("dur", 0)))
        starts[key] = [x["ts"] for x in evs]

    linked = 0
    for k in events:
        if k.get("cat") != "kernel":
            continue
        args = k.get("args") or {}
        if "External id" in args:
            continue
        launch = launches.get(args.get("correlation"))
        if launch is None:
            continue
        key = (launch.get("pid"), launch.get("tid"))
        evs = by_thread.get(key)
        if not evs:
            continue
        t = launch["ts"]
        i = bisect.bisect_right(starts[key], t) - 1
        # Scopes are properly nested, so the innermost enclosing one is the
        # latest-starting event that still contains t.
        while i >= 0:
            c = evs[i]
            if c["ts"] <= t <= c["ts"] + c.get("dur", 0):
                args["External id"] = c["args"]["External id"]
                k["args"] = args
                linked += 1
                break
            i -= 1
    return linked
