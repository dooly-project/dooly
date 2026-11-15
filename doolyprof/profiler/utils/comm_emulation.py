"""
Emulated communication for single-GPU tracing of a multi-rank deployment.

A fake parallel group is a real vLLM ``GroupCoordinator``, built by the engine's
own ``init_model_parallel_group`` for N ranks with this process as rank 0. Only
the transport underneath it is emulated:

* every ``torch.distributed`` process group the engine creates for it is an
  ``EmulatedProcessGroup``; ``torch.distributed`` calls on such a group are
  no-ops (``install_torch_distributed_passthrough``);
* the platform device communicator (``current_platform.get_device_communicator_cls()``)
  is the engine's own class with the attributes its constructor would fill
  with a NCCL communicator (found in that constructor, ``nccl_transport_attrs``)
  holding vLLM's own ``PyNcclCommunicator``, built over a NCCL library whose
  entry points are no-ops. Every other optional transport (custom all-reduce,
  symmetric memory, ...) is absent, which the engine already handles.

Output buffers are therefore allocated, shaped and tainted by the engine's own
code for every operation it has (all_reduce, all_gather, all_gatherv,
reduce_scatter, reduce_scatterv, gather, broadcast, send, recv, dispatch,
combine, ...), whichever vLLM version is installed. Nothing here names an
operation or computes an output shape.

Recording: every public method of the device communicator (the engine's
``DeviceCommunicatorBase`` API and the platform's additions) that is called
with a payload (a tensor, or the shape of the tensor it will produce) is
wrapped in a ``COMM: <method> IN:[<payload taints>] ARGS:[<json>]`` profiler
range. Only the outermost call is recorded, so engine-internal fallbacks
(e.g. one primitive implemented with another) are not double counted. If the
call never reached the transport, a ``COMM_NOOP`` marker is emitted inside the
range and the parser drops it. ``torch.distributed`` calls on an emulated
group made outside any device-communicator method (e.g. composite helpers
that call ``torch.distributed`` directly) are recorded as zero-length
``COMM: <function>`` ranges with ``"api": "torch.distributed"``.

ARGS holds a Python literal (``repr``; read back with ``ast.literal_eval``) --
torch's chrome-trace export does not escape double quotes in range names, so
JSON would corrupt the trace file. It holds the method's bound arguments (``inspect.signature``):
tensors become ``{"tensor": i}``, tainted integers ``{"int": v, "taint": T}``,
shapes ``{"shape": [...]}``, dtypes ``{"dtype": name}``; plus ``api``,
``group`` (the engine's unique group name), ``world_size`` and the payload
``dtypes``. ``doolyprof.profiler.comm_args`` reads it back.
"""

import ast
import dataclasses
import functools
import importlib
import inspect
import types
import textwrap
import threading
from contextlib import contextmanager

import torch
from torch.profiler import record_function

COMM_PREFIX = "COMM: "
COMM_NOOP_MARKER = "COMM_NOOP"

_state = threading.local()


# ---------------------------------------------------------------------------
# Formatting helpers (taint strings in the format the parser already reads)
# ---------------------------------------------------------------------------
def format_tensor_shape(tensor):
    from doolyprof.tracer.tensor import TaintedTensor
    if isinstance(tensor, TaintedTensor):
        return tensor.taint_str_with_history()
    if isinstance(tensor, torch.Tensor):
        return "[" + ", ".join(f"?({d})" for d in tensor.shape) + "]"
    return None


def _dim_taint_of(x):
    from doolyprof.tracer.types import DimTaint
    t = getattr(x, "taint", None)
    if t is None or isinstance(t, DimTaint):
        return t
    return DimTaint(t)


def format_shape(shape) -> str:
    """Taint string for a shape passed in place of a tensor (e.g. ``recv(size, dtype)``)."""
    from doolyprof.tracer.tensor import _format_dim
    return "[" + ", ".join(_format_dim(int(d), _dim_taint_of(d), include_history=True) for d in shape) + "]"


def _taint_label(t):
    from doolyprof.tracer.types import DimTaint
    while isinstance(t, DimTaint):
        t = t.taint
    return None if t is None else str(t)


def _is_shape(obj) -> bool:
    from doolyprof.tracer.types import TaintedShape
    return isinstance(obj, (torch.Size, TaintedShape))


def _collect_tensors(obj, out):
    if isinstance(obj, torch.Tensor):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_tensors(v, out)
    elif isinstance(obj, (list, tuple)) and not _is_shape(obj):
        for v in obj:
            _collect_tensors(v, out)


def _jsonable(obj, tensors):
    from doolyprof.tracer.types import TaintedInt
    if isinstance(obj, torch.Tensor):
        tensors.append(obj)
        return {"tensor": len(tensors) - 1}
    if isinstance(obj, TaintedInt):
        label = _taint_label(obj.taint)
        return {"int": obj.value, "taint": label} if label is not None else obj.value
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if _is_shape(obj):
        return {"shape": [_jsonable(d, tensors) for d in obj]}
    if isinstance(obj, torch.dtype):
        return {"dtype": str(obj).replace("torch.", "")}
    if isinstance(obj, torch.device):
        return {"device": str(obj)}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v, tensors) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v, tensors) for k, v in obj.items()}
    return {"object": type(obj).__name__}


def _annotation(name, call_args, api, group_name, world_size):
    """(range name, has_payload) for a call with bound arguments ``call_args``."""
    tensors = []
    spec = {k: _jsonable(v, tensors) for k, v in call_args.items()}
    shapes = [format_tensor_shape(t) for t in tensors]
    if not shapes:
        shapes = [format_shape(v) for v in call_args.values() if _is_shape(v)]
    if not shapes:
        return None, False
    meta = {
        "api": api,
        "group": group_name,
        "world_size": world_size,
        "args": spec,
        "dtypes": [str(t.dtype).replace("torch.", "") for t in tensors],
    }
    text = f"{COMM_PREFIX}{name} IN:[{' | '.join(shapes)}] ARGS:[{encode_meta(meta)}]"
    return text, True


def encode_meta(meta) -> str:
    """Quote-safe literal for a trace range name: strings are plain identifiers,
    dtype names and taint labels here, so repr() uses single quotes only."""
    text = repr(meta)
    if '"' in text or "\\" in text:
        raise ValueError(f"COMM args cannot be written into a trace range name: {text}")
    return text


def note_transport():
    """Called by the emulated transports: the current recorded call did reach the wire."""
    frame = getattr(_state, "frame", None)
    if frame is not None:
        frame["hits"] += 1
    return frame is not None


# ---------------------------------------------------------------------------
# Emulated transports
# ---------------------------------------------------------------------------
class EmulatedProcessGroup:
    """Stand-in for a ``torch.distributed`` process group over ``ranks``; this
    process is global rank 0. ``torch.distributed`` calls on it are no-ops."""

    def __init__(self, ranks, group_name=""):
        self._ranks = list(ranks)
        self.group_name = group_name

    def rank(self):
        return self._ranks.index(0) if 0 in self._ranks else -1

    def size(self):
        return len(self._ranks)

    def __repr__(self):
        return f"EmulatedProcessGroup(ranks={self._ranks})"


class _CompletedWork:
    """What an ``async_op=True`` call on an emulated group returns."""

    def wait(self, *args, **kwargs):
        return True

    def is_completed(self):
        return True

    def is_success(self):
        return True

    def get_future(self):
        fut = torch.futures.Future()
        fut.set_result(None)
        return fut


class _NullNCCLLibrary:
    """Stand-in for vLLM's NCCL binding: every NCCL entry point is a no-op that
    returns 0 (ncclSuccess / a null handle). The engine allocates every output
    buffer before calling NCCL, so shapes are unaffected."""

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        if not name.startswith("nccl"):
            raise AttributeError(name)

        def _entry(*args, **kwargs):
            note_transport()
            return 0

        return _entry


def _bound_arguments(sig, args, kwargs, skip_first=False):
    try:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        items = list(bound.arguments.items())
    except TypeError:
        items = [(f"arg{i}", a) for i, a in enumerate(args)] + list(kwargs.items())
    return dict(items[1:] if skip_first else items)


def _passthrough(fn, sig):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            group = sig.bind_partial(*args, **kwargs).arguments.get("group")
        except TypeError:
            group = kwargs.get("group")
        if not isinstance(group, EmulatedProcessGroup):
            return fn(*args, **kwargs)
        call_args = _bound_arguments(sig, args, kwargs)
        payload = []
        for v in call_args.values():
            _collect_tensors(v, payload)
        if payload and not note_transport():
            # A transfer made outside any device-communicator method: record it here.
            text, _ = _annotation(fn.__name__, {k: v for k, v in call_args.items() if k != "group"},
                                  "torch.distributed", group.group_name, group.size())
            if text:
                with record_function(text):
                    pass
        return _CompletedWork() if call_args.get("async_op") else None

    wrapper._dooly_passthrough = True
    return wrapper


def install_torch_distributed_passthrough():
    """Make every ``torch.distributed`` function that takes a ``group`` a no-op
    when that group is an ``EmulatedProcessGroup``; other groups are untouched.
    The function set comes from torch's own signatures."""
    import torch.distributed as dist
    from torch.distributed import distributed_c10d as c10d
    if getattr(c10d, "_dooly_passthrough_installed", False):
        return
    for name, fn in list(vars(c10d).items()):
        # type(), not isinstance(): touching deprecated module attributes warns
        if name.startswith("_") or type(fn) is not types.FunctionType or getattr(fn, "_dooly_passthrough", False):
            continue
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            continue
        if "group" not in sig.parameters:
            continue
        wrapped = _passthrough(fn, sig)
        setattr(c10d, name, wrapped)
        if getattr(dist, name, None) is fn:
            setattr(dist, name, wrapped)
    c10d._dooly_passthrough_installed = True


# ---------------------------------------------------------------------------
# Emulated device communicator
# ---------------------------------------------------------------------------
def import_qualname(qualname: str):
    module, _, attr = qualname.rpartition(".")
    return getattr(importlib.import_module(module), attr)


def _instantiate_dataclass(cls, **known):
    """Construct dataclass ``cls`` from ``known`` fields; required fields not in
    ``known`` get None (so the call follows the installed version's field list)."""
    kwargs = dict(known)
    for f in dataclasses.fields(cls):
        if not f.init or f.name in kwargs:
            continue
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            kwargs[f.name] = None
    return cls(**kwargs)


@contextmanager
def swap_attr(obj, name, value):
    had = name in vars(obj)
    old = vars(obj).get(name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if had:
            setattr(obj, name, old)
        else:
            delattr(obj, name)


def _function_local_imports(tree, module_name):
    """Names bound by ``from X import Y`` statements inside a function body."""
    names = {}
    package = module_name.rpartition(".")[0]
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module is None and not node.level:
            continue
        try:
            if node.level:
                base = package.rsplit(".", node.level - 1)[0] if node.level > 1 else package
                mod = importlib.import_module(f"{base}.{node.module}" if node.module else base)
            else:
                mod = importlib.import_module(node.module)
        except ImportError:
            continue
        for alias in node.names:
            if hasattr(mod, alias.name):
                names[alias.asname or alias.name] = getattr(mod, alias.name)
    return names


def nccl_transport_attrs(platform_cls):
    """Attributes the platform communicator's constructor assigns a vLLM
    ``PyNcclCommunicator`` to (read from the constructor's own source)."""
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    names = []
    for klass in platform_cls.__mro__:
        init = vars(klass).get("__init__")
        if init is None or klass is object:
            continue
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(init)))
        except (OSError, TypeError, SyntaxError):
            continue
        scope = dict(init.__globals__)
        scope.update(_function_local_imports(tree, init.__module__))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)) or not isinstance(node.value, ast.Call):
                continue
            func = node.value.func
            target_cls = scope.get(func.id) if isinstance(func, ast.Name) else None
            if not (isinstance(target_cls, type) and issubclass(target_cls, PyNcclCommunicator)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                    if t.attr not in names:
                        names.append(t.attr)
    return names


def null_pynccl(world_size: int, device):
    """vLLM's own PyNcclCommunicator for ``world_size`` ranks (this process is
    rank 0), constructed through its own ``__init__`` over a no-op NCCL library."""
    from vllm.distributed.device_communicators import pynccl as pynccl_mod
    from vllm.distributed.utils import StatelessProcessGroup
    group = _instantiate_dataclass(StatelessProcessGroup, rank=0, world_size=world_size,
                                   store=torch.distributed.HashStore())
    with swap_attr(pynccl_mod, "NCCLLibrary", _NullNCCLLibrary):
        return pynccl_mod.PyNcclCommunicator(group=group, device=device)


def _absent(self, name):
    # Attributes the platform constructor would set (optional transports,
    # feature flags) and that the emulated communicator does not have.
    if name.startswith("__"):
        raise AttributeError(name)
    return None


def _recording(name, method):
    sig = inspect.signature(method)

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if getattr(_state, "frame", None) is not None:
            return method(self, *args, **kwargs)
        call_args = _bound_arguments(sig, (self,) + args, kwargs, skip_first=True)
        text, has_payload = _annotation(name, call_args, "device_communicator",
                                        getattr(self, "unique_name", ""), getattr(self, "world_size", None))
        if not has_payload:
            return method(self, *args, **kwargs)
        _state.frame = {"hits": 0}
        try:
            with record_function(text):
                try:
                    return method(self, *args, **kwargs)
                finally:
                    if _state.frame["hits"] == 0:  # never reached the transport (or raised first)
                        with record_function(COMM_NOOP_MARKER):
                            pass
        finally:
            _state.frame = None

    return wrapper


def _public_methods(platform_cls):
    from vllm.distributed.device_communicators.base_device_communicator import DeviceCommunicatorBase
    names = []
    for klass in platform_cls.__mro__:
        if not (isinstance(klass, type) and issubclass(klass, DeviceCommunicatorBase)):
            continue
        for name, value in vars(klass).items():
            if not name.startswith("_") and inspect.isfunction(value) and name not in names:
                names.append(name)
    return names


_EMULATED_CLASSES = {}


def emulated_communicator_class(platform_cls):
    cls = _EMULATED_CLASSES.get(platform_cls)
    if cls is None:
        ns = {"__getattr__": _absent, "__module__": __name__}
        for name in _public_methods(platform_cls):
            ns[name] = _recording(name, getattr(platform_cls, name))
        cls = type(f"Emulated{platform_cls.__name__}", (platform_cls,), ns)
        _EMULATED_CLASSES[platform_cls] = cls
    return cls


def make_emulated_communicator(platform_cls, cpu_group, device=None, device_group=None, unique_name="", **extra):
    from vllm.distributed.device_communicators.base_device_communicator import DeviceCommunicatorBase
    comm = object.__new__(emulated_communicator_class(platform_cls))
    ranks = list(cpu_group._ranks)
    kwargs = dict(cpu_group=cpu_group, device=device, device_group=device_group, unique_name=unique_name,
                  global_ranks=ranks, global_world_size=max(ranks) + 1, **extra)
    params = inspect.signature(DeviceCommunicatorBase.__init__).parameters
    DeviceCommunicatorBase.__init__(comm, **{k: v for k, v in kwargs.items() if k in params})
    for attr in nccl_transport_attrs(platform_cls):
        setattr(comm, attr, null_pynccl(comm.world_size, comm.device))
    return comm


class EmulatedDeviceCommunicator:
    """Resolved by ``GroupCoordinator.__init__`` (through the platform's
    ``get_device_communicator_cls``) while an emulated group is built; returns
    an emulated instance of the platform's real communicator class."""

    platform_cls = None

    def __new__(cls, cpu_group, device=None, device_group=None, unique_name="", **extra):
        return make_emulated_communicator(cls.platform_cls, cpu_group, device, device_group, unique_name, **extra)


@contextmanager
def _swap_classmethod(owner, name, func):
    had = name in vars(owner)
    old = vars(owner).get(name)
    setattr(owner, name, classmethod(func))
    try:
        yield
    finally:
        if had:
            setattr(owner, name, old)
        else:
            delattr(owner, name)


def build_emulated_group(world_size: int, group_name: str, use_custom_op_call: bool = False):
    """A vLLM GroupCoordinator over ranks 0..world_size-1 (this process is rank 0),
    built by the engine's ``init_model_parallel_group`` on emulated transports.

    ``use_custom_op_call`` is turned off so the engine's methods call the device
    communicator directly in Python with the caller's (tainted) tensors, instead
    of routing them through ``torch.ops.vllm`` custom ops."""
    import vllm.distributed.parallel_state as ps
    from vllm.platforms import current_platform

    install_torch_distributed_passthrough()
    EmulatedDeviceCommunicator.platform_cls = import_qualname(current_platform.get_device_communicator_cls())
    world = ps.get_world_group()
    backend = torch.distributed.get_backend(world.device_group)
    qualname = f"{EmulatedDeviceCommunicator.__module__}.{EmulatedDeviceCommunicator.__qualname__}"

    def new_group(ranks=None, *args, **kwargs):
        return EmulatedProcessGroup(ranks if ranks is not None else range(world_size), group_name)

    with swap_attr(torch.distributed, "new_group", new_group), \
            _swap_classmethod(type(current_platform), "get_device_communicator_cls", lambda cls: qualname):
        group = ps.init_model_parallel_group(
            group_ranks=[list(range(world_size))],
            local_rank=world.local_rank,
            backend=backend,
            group_name=group_name,
        )
    group.use_custom_op_call = use_custom_op_call
    return group
