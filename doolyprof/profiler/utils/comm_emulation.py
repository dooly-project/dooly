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
range. A call made inside another recorded call (one primitive implemented
with another, a dispatch gathering over a group) is recorded as itself, and
transport calls count for the innermost recorded call only: a call that never
reached the transport itself -- an outer call whose data moved through inner
ones, or a call that moved nothing -- gets a ``COMM_NOOP`` marker inside its
range and the parser drops it, so nothing is counted twice. ``torch.distributed`` calls on an emulated
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
import re
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
# "COMM_RECV_ONLY:[i, ...]" inside a COMM range: the recorded tensors the call
# only received into (receive_only); they are not counted in its payload size.
COMM_RECV_ONLY_PREFIX = "COMM_RECV_ONLY:"

_state = threading.local()


# ---------------------------------------------------------------------------
# Formatting helpers (taint strings in the format the parser already reads)
# ---------------------------------------------------------------------------
def format_tensor_shape(tensor):
    from doolyprof.tracer.tensor import TaintedTensor
    if isinstance(tensor, TaintedTensor):
        return tensor.taint_str_with_history()
    if isinstance(tensor, torch.Tensor):
        # a plain tensor (a communication inside a custom op the tracer runs
        # as a black box): label its dims the way module annotations do, from
        # the enclosing op's inputs, then the registry
        from doolyprof.tracer.tensor import _format_dim
        dts = getattr(tensor, "_dim_taints", None)
        return "[" + ", ".join(
            _format_dim(int(d), dts[i] if dts is not None and i < len(dts) else None, include_history=True)
            for i, d in enumerate(torch.Tensor.size(tensor))) + "]"
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


_DIM_TEXT = re.compile(r"([A-Za-z_?]+(?:[+-]\d+)?)\((\d+)\)(\[hist:[^\]]*\])?")
WORLD_SIZE_LABEL = "WORLD_SIZE"


def label_gathered_dims(shapes, world_size):
    """Shape strings of one collective with each unexplained dim (``?(S)``)
    whose size is ``world_size`` times a NUM_TOKS / NUM_REQS dim of the same
    call relabelled ``MIX(S)[hist:<label>:<size>,WORLD_SIZE:<world_size>]``.

    Such a dim holds that batch dim from every rank of the group: an EP
    dispatch gathers each data-parallel rank's tokens, and the expert output
    its combine reduce-scatters back has as many rows. Every emulated rank
    runs this rank's batch, so the gathered size is the batch size times the
    group size. Those tensors live inside the MoE custom op, where the tracer
    sees plain tensors, so nothing else labels the gathered size (Qwen1.5-MoE
    on DP2: ``?(144)`` for 2 x 72 tokens). A size that two labels would
    explain is left as it is."""
    from doolyprof.tracer.types import split_offset_label

    w = int(world_size or 0)
    if w < 2:
        return shapes
    gathered = {}
    for text in shapes:
        for m in _DIM_TEXT.finditer(text or ""):
            parts = split_offset_label(m.group(1))
            if parts is not None and parts[1] == 0 and not parts[0].startswith("MAX_"):
                gathered.setdefault(int(m.group(2)) * w, set()).add((parts[0], int(m.group(2))))

    def relabel(m):
        size = int(m.group(2))
        if m.group(1) != "?" or len(gathered.get(size, ())) != 1:
            return m.group(0)
        label, per_rank = next(iter(gathered[size]))
        return f"MIX({size})[hist:{label}:{per_rank},{WORLD_SIZE_LABEL}:{w}]"

    return [_DIM_TEXT.sub(relabel, text) if text else text for text in shapes]


def _annotation(name, call_args, api, group_name, world_size, tensors=None):
    """(range name, has_payload) for a call with bound arguments ``call_args``;
    the recorded tensors, in ``{"tensor": i}`` order, are appended to ``tensors``."""
    tensors = [] if tensors is None else tensors
    spec = {k: _jsonable(v, tensors) for k, v in call_args.items()}
    shapes = label_gathered_dims([format_tensor_shape(t) for t in tensors], world_size)
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


# Buffer arguments of NCCL's C API (positions of sendbuff / recvbuff): which
# memory a call reads from and which it writes into.
_NCCL_BUFFERS = {
    "ncclAllReduce": (0, 1),
    "ncclAllGather": (0, 1),
    "ncclReduceScatter": (0, 1),
    "ncclBroadcast": (0, 1),
    "ncclReduce": (0, 1),
    "ncclSend": (0, None),
    "ncclRecv": (None, 0),
}


def note_buffers(nccl_name, args):
    """Called by the emulated NCCL library: the send / receive buffer addresses
    of one NCCL call made by the current recorded call."""
    frame = getattr(_state, "frame", None)
    send_recv = _NCCL_BUFFERS.get(nccl_name)
    if frame is None or send_recv is None:
        return
    for key, idx in zip(("send", "recv"), send_recv):
        if idx is not None and idx < len(args):
            ptr = _n(args[idx])
            if ptr:
                frame[key].append(ptr)


def receive_only(tensors, frame):
    """Indices of the recorded tensors the transport only wrote into (a
    caller-allocated output, e.g. reduce_scatterv_into_output's ``output``):
    a receive buffer is not part of the data the call sends."""
    def holds(t, ptrs):
        start = t.data_ptr()
        end = start + t.numel() * t.element_size()
        return any(start <= p < end for p in ptrs)
    return [i for i, t in enumerate(tensors)
            if t.numel() and holds(t, frame["recv"]) and not holds(t, frame["send"])]


# ---------------------------------------------------------------------------
# Emulated transports
# ---------------------------------------------------------------------------
class _CallableInt(int):
    """An int that can also be called: torch's ``ProcessGroup`` has ``rank()``,
    vLLM's ``StatelessProcessGroup`` has a ``rank`` attribute."""

    def __call__(self):
        return int(self)


class EmulatedProcessGroup:
    """Stand-in for a process group over ``ranks``; this process is global rank 0.
    ``torch.distributed`` calls on it are no-ops. It also answers the API of
    vLLM's ``StatelessProcessGroup`` (``rank``/``world_size`` attributes,
    ``broadcast_obj``, ``barrier``, ...), which vLLM code takes for any group
    that is not a torch ``ProcessGroup``: objects exchanged through it come back
    as this rank's own (every rank is assumed to hold the same)."""

    def __init__(self, ranks, group_name=""):
        self._ranks = list(ranks)
        self.group_name = group_name

    @property
    def rank(self):
        return _CallableInt(self._ranks.index(0) if 0 in self._ranks else -1)

    def size(self):
        return len(self._ranks)

    @property
    def world_size(self):
        return len(self._ranks)

    def broadcast_obj(self, obj, src=None):
        return obj

    def all_gather_obj(self, obj):
        return [obj] * len(self._ranks)

    def barrier(self, *args, **kwargs):
        return None

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


_CUDART = None
_NCCL_DTYPE_BYTES = None


def _device_copy(dst, src, nbytes, stream):
    """Asynchronous device copy of ``nbytes`` from ``src`` to ``dst`` (raw
    pointers, as NCCL receives them) on ``stream``, through the CUDA runtime vLLM
    already loaded."""
    global _CUDART
    import ctypes
    dst, src = getattr(dst, "value", dst), getattr(src, "value", src)
    if not nbytes or not dst or not src or dst == src:
        return
    if _CUDART is None:
        from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
        lib = CudaRTLibrary().lib
        fn = lib.cudaMemcpyAsync
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
        fn.restype = ctypes.c_int
        _CUDART = fn
    _CUDART(dst, src, nbytes, 4, getattr(stream, "value", stream))  # 4 = cudaMemcpyDefault


def _nccl_dtype_bytes(datatype):
    """Element size of an ``ncclDataType_t``, from vLLM's own torch <-> NCCL mapping."""
    global _NCCL_DTYPE_BYTES
    if _NCCL_DTYPE_BYTES is None:
        from vllm.distributed.device_communicators.pynccl_wrapper import ncclDataTypeEnum
        _NCCL_DTYPE_BYTES = {}
        for name in ("int8", "uint8", "int32", "uint32", "int64", "uint64", "float16", "float32",
                     "float64", "bfloat16", "float8_e4m3fn", "float8_e5m2"):
            dtype = getattr(torch, name, None)
            try:
                _NCCL_DTYPE_BYTES[int(ncclDataTypeEnum.from_torch(dtype))] = torch.empty((), dtype=dtype).element_size()
            except Exception:  # noqa: BLE001 - dtype this NCCL binding does not map
                continue
    return _NCCL_DTYPE_BYTES[int(getattr(datatype, "value", datatype))]


def _n(x):
    """Plain int from a ctypes value or an int."""
    return int(getattr(x, "value", x) or 0)


# What each data-moving NCCL primitive (C API argument order) leaves in rank 0's
# receive buffer when every rank sent what rank 0 sent. Values are not reduced:
# the copies only make received data valid for code that reads it (expert ids).
def _move_all_reduce(world, send, recv, count, datatype, op, comm, stream):
    _device_copy(recv, send, _n(count) * _nccl_dtype_bytes(datatype), stream)


def _move_all_gather(world, send, recv, count, datatype, comm, stream):
    nbytes = _n(count) * _nccl_dtype_bytes(datatype)
    for r in range(world):
        _device_copy(_n(recv) + r * nbytes, send, nbytes, stream)


def _move_reduce_scatter(world, send, recv, count, datatype, op, comm, stream):
    _device_copy(recv, send, _n(count) * _nccl_dtype_bytes(datatype), stream)


def _move_broadcast(world, send, recv, count, datatype, root, comm, stream):
    _device_copy(recv, send, _n(count) * _nccl_dtype_bytes(datatype), stream)


def _move_reduce(world, send, recv, count, datatype, op, root, comm, stream):
    if _n(root) == 0:
        _device_copy(recv, send, _n(count) * _nccl_dtype_bytes(datatype), stream)


_SYMMETRIC_MOVES = {
    "ncclAllReduce": _move_all_reduce,
    "ncclAllGather": _move_all_gather,
    "ncclReduceScatter": _move_reduce_scatter,
    "ncclBroadcast": _move_broadcast,
    "ncclReduce": _move_reduce,
}


class _NullNCCLLibrary:
    """Stand-in for vLLM's NCCL binding: every NCCL entry point is a no-op that
    returns 0 (ncclSuccess / a null handle). The engine allocates every output
    buffer before calling NCCL, so shapes are unaffected.

    With ``replicate`` (groups whose received data feeds data-dependent code,
    e.g. the expert ids an EP dispatch gathers from every DP rank), the
    data-moving primitives fill rank 0's receive buffer as if every rank had sent
    what rank 0 sent, instead of leaving it uninitialized."""

    def __init__(self, *args, world_size=1, replicate=False, **kwargs):
        self._world_size = world_size
        self._replicate = replicate

    def __getattr__(self, name):
        if not name.startswith("nccl"):
            raise AttributeError(name)
        move = _SYMMETRIC_MOVES.get(name) if self._replicate else None

        def _entry(*args, **kwargs):
            note_transport()
            note_buffers(name, args)
            if move is not None:
                move(self._world_size, *args, **kwargs)
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


def null_pynccl(world_size: int, device, replicate: bool = False):
    """vLLM's own PyNcclCommunicator for ``world_size`` ranks (this process is
    rank 0), constructed through its own ``__init__`` over a no-op NCCL library
    (``replicate``: see ``_NullNCCLLibrary``)."""
    from vllm.distributed.device_communicators import pynccl as pynccl_mod
    from vllm.distributed.utils import StatelessProcessGroup
    group = _instantiate_dataclass(StatelessProcessGroup, rank=0, world_size=world_size,
                                   store=torch.distributed.HashStore())
    library = functools.partial(_NullNCCLLibrary, world_size=world_size, replicate=replicate)
    with swap_attr(pynccl_mod, "NCCLLibrary", library):
        return pynccl_mod.PyNcclCommunicator(group=group, device=device)


def _all2all_setup_blocks(platform_cls):
    """The ``if self.use_all2all:`` blocks of the platform communicator's
    constructor (the engine's own selection of its all2all manager for the
    configured backend), with the scope their code runs in."""
    blocks = []
    for klass in platform_cls.__mro__:
        init = vars(klass).get("__init__")
        if init is None or klass is object:
            continue
        try:
            source = textwrap.dedent(inspect.getsource(init))
            tree = ast.parse(source)
        except (OSError, TypeError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            test = ast.unparse(node.test)
            if "use_all2all" not in test:
                continue
            assigns = any(isinstance(t, ast.Attribute) and t.attr == "all2all_manager"
                          for sub in ast.walk(node) if isinstance(sub, ast.Assign) for t in sub.targets)
            if assigns:
                blocks.append((init, node))
        if blocks:
            break
    return blocks


def install_all2all_manager(platform_cls, comm, **constructor_args):
    """Give an emulated EP communicator the all2all manager the platform
    constructor would build for the configured backend, by running that part of
    the constructor's own code on it (the rest of the constructor builds real
    transports, which the emulated communicator does not have)."""
    if getattr(comm, "all2all_manager", None) is not None:
        return
    for init, node in _all2all_setup_blocks(platform_cls):
        scope = dict(init.__globals__)
        scope.update(_function_local_imports(ast.parse(textwrap.dedent(inspect.getsource(init))), init.__module__))
        for name, param in inspect.signature(init).parameters.items():
            if param.default is not inspect.Parameter.empty:
                scope[name] = param.default
        scope.update(constructor_args)
        scope["self"] = comm
        exec(compile(ast.Module(body=[node], type_ignores=[]), inspect.getsourcefile(init) or "<init>", "exec"), scope)


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
        outer = getattr(_state, "frame", None)
        # A call that moves its data through another recorded call -- on another
        # group's communicator (an EP dispatch gathering over the DP group) or on
        # this one (a sequence-parallel dispatch gathering over the EP group
        # itself, one primitive implemented with another) -- is recorded as that
        # inner call, which is what the transport runs and what a real group can
        # replay. Transport calls count for the innermost recorded call only, so
        # the outer call, which then never reaches the transport itself, is
        # marked COMM_NOOP and nothing is counted twice.
        call_args = _bound_arguments(sig, (self,) + args, kwargs, skip_first=True)
        tensors = []
        text, has_payload = _annotation(name, call_args, "device_communicator",
                                        getattr(self, "unique_name", ""), getattr(self, "world_size", None),
                                        tensors)
        if not has_payload:
            return method(self, *args, **kwargs)
        frame = _state.frame = {"hits": 0, "send": [], "recv": []}
        try:
            with record_function(text):
                try:
                    return method(self, *args, **kwargs)
                finally:
                    if frame["hits"] == 0:  # never reached the transport (or raised first)
                        with record_function(COMM_NOOP_MARKER):
                            pass
                    else:
                        recv_only = receive_only(tensors, frame)
                        if recv_only:
                            with record_function(f"{COMM_RECV_ONLY_PREFIX}{recv_only}"):
                                pass
        finally:
            _state.frame = outer

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


def make_emulated_communicator(platform_cls, cpu_group, device=None, device_group=None, unique_name="",
                               replicate=False, use_all2all=False, **extra):
    from vllm.distributed.device_communicators.base_device_communicator import DeviceCommunicatorBase
    comm = object.__new__(emulated_communicator_class(platform_cls))
    ranks = list(cpu_group._ranks)
    kwargs = dict(cpu_group=cpu_group, device=device, device_group=device_group, unique_name=unique_name,
                  global_ranks=ranks, global_world_size=max(ranks) + 1, **extra)
    params = inspect.signature(DeviceCommunicatorBase.__init__).parameters
    DeviceCommunicatorBase.__init__(comm, **{k: v for k, v in kwargs.items() if k in params})
    for attr in nccl_transport_attrs(platform_cls):
        setattr(comm, attr, null_pynccl(comm.world_size, comm.device, replicate=replicate))
    if use_all2all:
        # Engines that do not pass the flag derive it from their own
        # configuration, which runs one data-parallel rank here; some also read
        # the all2all backend only then (or only inside an engine config scope).
        comm.use_all2all = True
        if getattr(comm, "all2all_backend", None) is None:
            from vllm.config import get_current_vllm_config_or_none
            config = get_current_vllm_config_or_none()
            if config is not None:
                comm.all2all_backend = config.parallel_config.all2all_backend
        install_all2all_manager(platform_cls, comm, **{k: v for k, v in extra.items()
                                                       if k in inspect.signature(platform_cls.__init__).parameters})
    return comm


class EmulatedDeviceCommunicator:
    """Resolved by ``GroupCoordinator.__init__`` (through the platform's
    ``get_device_communicator_cls``) while an emulated group is built; returns
    an emulated instance of the platform's real communicator class."""

    platform_cls = None
    replicate = False
    use_all2all = False

    def __new__(cls, cpu_group, device=None, device_group=None, unique_name="", **extra):
        extra.pop("use_all2all", None)
        return make_emulated_communicator(cls.platform_cls, cpu_group, device, device_group, unique_name,
                                          replicate=cls.replicate, use_all2all=cls.use_all2all, **extra)


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


def build_emulated_group(world_size: int, group_name: str, use_custom_op_call: bool = False,
                         use_all2all: bool = False, replicate: bool = False):
    """A vLLM GroupCoordinator over ranks 0..world_size-1 (this process is rank 0),
    built by the engine's ``init_model_parallel_group`` on emulated transports.

    ``use_custom_op_call`` is turned off so the engine's methods call the device
    communicator directly in Python with the caller's (tainted) tensors, instead
    of routing them through ``torch.ops.vllm`` custom ops. ``use_all2all``: the
    group is an expert-parallel group that dispatches/combines through the
    engine's all2all manager. ``replicate``: data received on the group is this
    rank's own (``_NullNCCLLibrary``)."""
    import vllm.distributed.parallel_state as ps
    from vllm.platforms import current_platform

    install_torch_distributed_passthrough()
    platform_cls = import_qualname(current_platform.get_device_communicator_cls())
    if platform_cls is EmulatedDeviceCommunicator:
        # Built while another emulated group is being built (an EP group's
        # all2all manager looks up the DP group), when the platform resolves to
        # the stand-in: use the platform class the outer build resolved.
        platform_cls = EmulatedDeviceCommunicator.platform_cls
    world = ps.get_world_group()
    backend = torch.distributed.get_backend(world.device_group)
    qualname = f"{EmulatedDeviceCommunicator.__module__}.{EmulatedDeviceCommunicator.__qualname__}"

    def new_group(ranks=None, *args, **kwargs):
        return EmulatedProcessGroup(ranks if ranks is not None else range(world_size), group_name)

    options = {}
    if "use_all2all" in inspect.signature(ps.init_model_parallel_group).parameters:
        options["use_all2all"] = use_all2all
    # A group can be built while another is (an EP group's all2all manager looks
    # up the DP group): each build sees its own options.
    saved = (EmulatedDeviceCommunicator.platform_cls, EmulatedDeviceCommunicator.replicate,
             EmulatedDeviceCommunicator.use_all2all)
    EmulatedDeviceCommunicator.platform_cls = platform_cls
    EmulatedDeviceCommunicator.replicate = replicate
    EmulatedDeviceCommunicator.use_all2all = use_all2all
    try:
        with swap_attr(torch.distributed, "new_group", new_group), \
                _swap_classmethod(type(current_platform), "get_device_communicator_cls", lambda cls: qualname):
            group = ps.init_model_parallel_group(
                group_ranks=[list(range(world_size))],
                local_rank=world.local_rank,
                backend=backend,
                group_name=group_name,
                **options,
            )
    finally:
        (EmulatedDeviceCommunicator.platform_cls, EmulatedDeviceCommunicator.replicate,
         EmulatedDeviceCommunicator.use_all2all) = saved
    group.use_custom_op_call = use_custom_op_call
    return group
