"""Unit tests for Triton launch scopes (tracer/triton_launch.py, the parser's
TRITON anchors and profiler/utils/triton_replay.py).

vLLM 0.33 Kimi-K3 launches its attention-residual kernel straight from the
model's forward; the kernel anchored to the whole model and folded every layer
into one unit that cannot run alone. A replayable launch now gets its own
scope and is relaunched from the recorded spec.

Runs directly: ``python doolyprof/tests/triton_scope_test.py``. CPU only.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import torch  # noqa: E402

from doolyprof.tracer.hooks import install_hooks  # noqa: E402

install_hooks()

from doolyprof.tracer import enable_taint_tracking  # noqa: E402
from doolyprof.tracer.tensor import TaintedTensor  # noqa: E402
from doolyprof.tracer.triton_launch import launch_annotation  # noqa: E402
from doolyprof.tracer.types import Taint, TaintedInt  # noqa: E402

enable_taint_tracking()


def _attn_res_kernel(*a, **k):  # stands in for the kernel's Python function
    pass


class _Jit:
    fn = _attn_res_kernel


LAUNCHES = []


class FakeKernel:
    """Relaunch target: records the grid and arguments it is launched with."""
    def __getitem__(self, grid):
        return lambda *args, **kwargs: LAUNCHES.append((grid, args, kwargs))


fake_kernel = FakeKernel()


def _tok(n=59, h=64):
    return TaintedTensor(torch.zeros(n, h), (Taint("NUM_TOKS"), Taint("MODEL_CONFIG")))


def _attn_res_launch(prefix, blocks, weight, num_tokens, extra=()):
    # _attn_res_kernel[(num_tokens,)](prefix, None, blocks, weight, out, stride, n_blocks, ...)
    out = prefix.new_empty(prefix.shape)
    args = (prefix, None, blocks, weight, out, 64, 8, *extra)
    kwargs = {"grid": (num_tokens,), "warmup": False, "HAS_DELTA": False, "BLOCK_D": 64, "num_warps": 4}
    return args, kwargs


def _spec(name):
    from doolyprof.tracer.triton_launch import decode_spec
    assert "\"" not in name, "the saved trace rewrites double quotes"
    return decode_spec(name.split(" SPEC:", 1)[1])


def test_attn_res_launch_gets_a_replayable_scope():
    prefix = _tok()
    blocks = TaintedTensor(torch.zeros(59, 8, 64), (Taint("NUM_TOKS"), Taint("MODEL_CONFIG"), Taint("MODEL_CONFIG")))
    weight = torch.zeros(64)
    args, kwargs = _attn_res_launch(prefix, blocks, weight, prefix.size(0))
    name = launch_annotation(_Jit(), args, kwargs)
    assert name is not None and name.startswith(f"TRITON: {__name__}:_attn_res_kernel IN:[[NUM_TOKS(59)"), name
    spec = _spec(name)
    assert spec["grid"] == [{"dim": [0, 0]}], spec["grid"]
    assert spec["args"][:3] == [{"t": 0}, {"v": None}, {"t": 1}], spec["args"]
    assert spec["args"][5:] == [{"v": 64}, {"v": 8}], spec["args"]
    assert spec["kwargs"] == {"HAS_DELTA": {"v": False}, "BLOCK_D": {"v": 64}, "num_warps": {"v": 4}}, spec["kwargs"]
    # a different constexpr is a different kernel: the state hash differs
    args2, kwargs2 = _attn_res_launch(prefix, blocks, weight, prefix.size(0))
    kwargs2["HAS_DELTA"] = True
    other = launch_annotation(_Jit(), args2, kwargs2)
    state = lambda n: n.split(" STATE:")[1].split(" ")[0]  # noqa: E731
    assert state(name) != state(other)


def test_launches_that_cannot_be_replayed_get_no_scope():
    prefix = _tok()
    blocks = _tok()
    weight = torch.zeros(64)
    # an index tensor (sequence boundaries, state slots) needs engine metadata
    idx = TaintedTensor(torch.zeros(60, dtype=torch.int32), (Taint("NUM_REQS"),))
    args, kwargs = _attn_res_launch(prefix, blocks, weight, prefix.size(0), extra=(idx,))
    assert launch_annotation(_Jit(), args, kwargs) is None
    # a grid function cannot be replayed
    args, kwargs = _attn_res_launch(prefix, blocks, weight, prefix.size(0))
    kwargs["grid"] = lambda meta: (59,)
    assert launch_annotation(_Jit(), args, kwargs) is None
    # an untainted integer equal to the token count may be a batch size that lost its taint
    args, kwargs = _attn_res_launch(prefix, blocks, weight, prefix.size(0), extra=(59,))
    assert launch_annotation(_Jit(), args, kwargs) is None
    # a batch dim after a fixed dim makes plain strides batch dependent
    transposed = TaintedTensor(torch.zeros(64, 59), (Taint("MODEL_CONFIG"), Taint("NUM_TOKS")))
    args, kwargs = _attn_res_launch(prefix, transposed, weight, prefix.size(0))
    assert launch_annotation(_Jit(), args, kwargs) is None
    # plain tensors only (a launch inside a custom op the tracer runs as a black box)
    args, kwargs = _attn_res_launch(torch.zeros(59, 64), torch.zeros(59, 64), weight, 59)
    assert launch_annotation(_Jit(), args, kwargs) is None


def _trace_events(scope_name):
    """A module whose forward launches one kernel through a TRITON scope and
    runs one op of its own (the kernel's External id is the scope's)."""
    return [
        {"ph": "X", "cat": "user_annotation", "name": "MODULE: KimiLinearModel IN:[[NUM_TOKS(59)]] PATH:model",
         "ts": 0, "dur": 100, "pid": 1, "tid": 1, "args": {"External id": 1}},
        {"ph": "X", "cat": "cpu_op", "name": "aten::mm", "ts": 5, "dur": 10, "pid": 1, "tid": 1,
         "args": {"External id": 2}},
        {"ph": "X", "cat": "user_annotation", "name": "API: mm IN:[[NUM_TOKS(59), MODEL_CONFIG(64)]]",
         "ts": 4, "dur": 12, "pid": 1, "tid": 1, "args": {}},
        {"ph": "X", "cat": "user_annotation", "name": scope_name, "ts": 40, "dur": 20, "pid": 1, "tid": 1,
         "args": {"External id": 3}},
        {"ph": "X", "cat": "kernel", "name": "gemm", "ts": 200, "dur": 5, "pid": 0, "tid": 7,
         "args": {"External id": 2}},
        {"ph": "X", "cat": "kernel", "name": "_attn_res_kernel", "ts": 210, "dur": 5, "pid": 0, "tid": 7,
         "args": {"External id": 3}},
    ]


def test_parser_anchors_the_kernel_to_its_scope_not_the_module():
    from doolyprof.profiler.parser import HierarchicalTraceParser
    prefix, blocks, weight = _tok(), _tok(), torch.zeros(64)
    args, kwargs = _attn_res_launch(prefix, blocks, weight, prefix.size(0))
    scope = launch_annotation(_Jit(), args, kwargs)
    parser = HierarchicalTraceParser(events=_trace_events(scope))
    parser.parse_from_events()
    hierarchies = parser.extract_module_to_operation_mapping()
    anchors = {h.anchor.name: h for h in hierarchies}
    assert f"triton::{__name__}:_attn_res_kernel" in anchors, sorted(anchors)
    assert "KimiLinearModel" not in anchors, sorted(anchors)
    h = anchors[f"triton::{__name__}:_attn_res_kernel"]
    assert [k.name for k in h.kernels] == ["_attn_res_kernel"]
    assert h.anchor.args["triton_spec"]["grid"] == [{"dim": [0, 0]}]
    assert h.anchor.args["state_hash"]


def test_replay_reads_grid_and_batch_integers_off_the_tensors():
    from doolyprof.profiler.utils.triton_replay import TritonLaunchCall
    spec = {"fn": f"{__name__}:fake_kernel",
            "args": [{"t": 0}, {"v": None}, {"t": 1}, {"v": 64}, {"dim": [0, 0]}],
            "kwargs": {"HAS_DELTA": {"v": False}}, "grid": [{"dim": [0, 0]}]}
    call = TritonLaunchCall(spec)
    a, b = torch.zeros(16, 64), torch.zeros(16, 8, 64)
    LAUNCHES.clear()
    call(a, b)
    grid, args, kwargs = LAUNCHES[0]
    assert grid == (16,), grid
    assert args[0] is a and args[1] is None and args[2] is b and args[3:] == (64, 16), args
    assert kwargs == {"HAS_DELTA": False}


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except AssertionError as e:
                failures += 1; print(f"FAIL {name}: {e}")
            except Exception as e:  # noqa: BLE001
                import traceback; traceback.print_exc()
                failures += 1; print(f"ERROR {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failures else 0)
