"""Launch a traced Triton kernel again from its TRITON scope's SPEC.

``doolyprof.tracer.triton_launch`` records, for a launch made from Python with
tainted tensors, the kernel's import path and each argument as one of:
  {"t": i}        the i-th tensor argument (the profiler generates it from the
                  scope's IN shapes, in the same order)
  {"dim": [i, d]} a batch-dependent integer: dim d of the i-th tensor
  {"v": value}    a fixed value (config constant, constexpr, None, flag)
The grid is recorded the same way. ``TritonLaunchCall`` is called with the
generated tensors and launches the kernel with the same fixed values, the grid
and batch integers read off those tensors.
"""
import importlib


def _jit_function(path):
    module_name, _, attr = path.partition(":")
    obj = getattr(importlib.import_module(module_name), attr)
    # an @triton.autotune / @triton.heuristics wrapper holds the JITFunction
    # in .fn; the traced launch already carries the chosen configuration
    while not hasattr(obj, "run") or type(obj).__name__ != "JITFunction":
        inner = getattr(obj, "fn", None)
        if inner is None or inner is obj:
            break
        obj = inner
    return obj


class TritonLaunchCall:
    """Callable(*tensors) that relaunches one traced Triton kernel."""

    def __init__(self, spec):
        self.spec = spec
        self.kernel = _jit_function(spec["fn"])
        self.__name__ = spec["fn"]

    def _value(self, entry, tensors):
        if "t" in entry:
            return tensors[entry["t"]]
        if "dim" in entry:
            i, d = entry["dim"]
            return int(tensors[i].shape[d])
        return entry["v"]

    def __call__(self, *tensors):
        args = [self._value(a, tensors) for a in self.spec["args"]]
        kwargs = {k: self._value(v, tensors) for k, v in self.spec["kwargs"].items()}
        grid = tuple(self._value(g, tensors) for g in self.spec["grid"])
        return self.kernel[grid](*args, **kwargs)


def launch_callable(module):
    """The relaunch callable for a ModuleInfo of a Triton launch scope, else None."""
    spec = getattr(module, "launch_spec", None)
    return TritonLaunchCall(spec) if spec else None
