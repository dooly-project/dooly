"""Unit tests for tracer taint propagation through tensor ops that vLLM's model
code calls on the hot path.

* ``squeeze(dim=(a, b))`` must accept a tuple of dims (vLLM 0.32 Mamba
  ``bind_kv_cache`` raised ``'<' not supported between tuple and int``).
* Kernels launched through the CUDA driver API outside an aten op (Triton)
  get the External id of the innermost CPU event enclosing their launch.
* A non-``nn.Module`` engine component (vLLM 0.32 V2-runner ``Sampler``) gets a
  ``MODULE:`` annotation around its calls; nn.Modules are left alone.
* ``view(dtype)`` must keep the dimension taints (vLLM 0.32 ends
  ``VocabParallelEmbedding.forward`` with ``output.view(output_dtype)``; losing
  them untainted the first decoder layer's norm and QKV projection).

Runs directly: ``python doolyprof/tests/tracer_ops_test.py``. CPU only.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import torch  # noqa: E402

from doolyprof.tracer.hooks import install_hooks  # noqa: E402

install_hooks()

from doolyprof.tracer import enable_taint_tracking  # noqa: E402
from doolyprof.tracer.tensor import TaintedTensor  # noqa: E402
from doolyprof.tracer.types import Taint  # noqa: E402

enable_taint_tracking()


def _labels(t):
    return [None if d is None else str(d.taint) for d in t._dim_taints]


def _tokens_by_hidden(dtype):
    return TaintedTensor(torch.zeros(9, 64, dtype=dtype), (Taint("NUM_TOKS"), Taint("MODEL_CONFIG")))


def test_view_same_width_dtype_keeps_all_taints():
    out = _tokens_by_hidden(torch.bfloat16).view(torch.float16)
    assert isinstance(out, TaintedTensor), type(out)
    assert _labels(out) == ["NUM_TOKS", "MODEL_CONFIG"], _labels(out)


def test_view_round_trip_dtype_keeps_all_taints():
    x = _tokens_by_hidden(torch.bfloat16)
    out = x.view(x.dtype)
    assert isinstance(out, TaintedTensor), type(out)
    assert _labels(out) == ["NUM_TOKS", "MODEL_CONFIG"], _labels(out)


def test_view_narrower_dtype_drops_only_last_dim_taint():
    out = _tokens_by_hidden(torch.bfloat16).view(torch.int8)
    assert tuple(out.shape) == (9, 128), tuple(out.shape)
    assert isinstance(out, TaintedTensor), type(out)
    assert _labels(out) == ["NUM_TOKS", None], _labels(out)


def test_squeeze_tuple_dims():
    x = TaintedTensor(torch.zeros(5, 1, 1, 7), (Taint("NUM_REQS"), None, None, Taint("MODEL_CONFIG")))
    out = x.squeeze(dim=(1, 2))
    assert tuple(out.shape) == (5, 7), tuple(out.shape)
    assert _labels(out) == ["NUM_REQS", "MODEL_CONFIG"], _labels(out)
    out = torch.squeeze(x, (1, -1))  # -1 has size 7, so only dim 1 goes
    assert tuple(out.shape) == (5, 1, 7), tuple(out.shape)
    assert _labels(out) == ["NUM_REQS", None, "MODEL_CONFIG"], _labels(out)


def test_driver_launched_kernel_gets_enclosing_external_id():
    from doolyprof.tracer.kineto_links import attach_launch_external_ids
    ev = [
        {"ph": "X", "cat": "user_annotation", "name": "MODULE: KDA", "pid": 1, "tid": 7,
         "ts": 100, "dur": 100, "args": {"External id": 10}},
        {"ph": "X", "cat": "user_annotation", "name": "MODULE: Norm", "pid": 1, "tid": 7,
         "ts": 150, "dur": 20, "args": {"External id": 11}},
        {"ph": "X", "cat": "cpu_op", "name": "aten::mm", "pid": 1, "tid": 7,
         "ts": 120, "dur": 5, "args": {"External id": 12}},
        # driver launches: one directly under KDA, one inside the nested Norm
        {"ph": "X", "cat": "cuda_driver", "name": "cuLaunchKernelEx", "pid": 1, "tid": 7,
         "ts": 130, "dur": 2, "args": {"correlation": 500}},
        {"ph": "X", "cat": "cuda_driver", "name": "cuLaunchKernelEx", "pid": 1, "tid": 7,
         "ts": 160, "dur": 2, "args": {"correlation": 501}},
        {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "pid": 1, "tid": 7,
         "ts": 121, "dur": 1, "args": {"correlation": 502, "External id": 12}},
        {"ph": "X", "cat": "kernel", "name": "chunk_kda_fwd", "ts": 300, "dur": 9, "args": {"correlation": 500}},
        {"ph": "X", "cat": "kernel", "name": "layer_norm_gated", "ts": 310, "dur": 9, "args": {"correlation": 501}},
        {"ph": "X", "cat": "kernel", "name": "gemm", "ts": 290, "dur": 9, "args": {"correlation": 502, "External id": 12}},
    ]
    assert attach_launch_external_ids(ev) == 2
    by_name = {e["name"]: e["args"].get("External id") for e in ev if e["cat"] == "kernel"}
    assert by_name == {"chunk_kda_fwd": 10, "layer_norm_gated": 11, "gemm": 12}, by_name


def test_engine_callable_annotated_like_module():
    from doolyprof.tracer.hooks import annotate_engine_callable

    class Sampler:  # stands in for vllm.v1.worker.gpu.sample.sampler.Sampler
        def __call__(self, logits, input_batch=None):
            return logits.argmax(dim=-1)

    logits = TaintedTensor(torch.zeros(5, 32), (Taint("NUM_REQS"), Taint("MODEL_CONFIG")))
    sampler = Sampler()
    assert annotate_engine_callable(sampler)
    assert not annotate_engine_callable(Sampler())        # already wrapped
    assert not annotate_engine_callable(torch.nn.Linear(2, 2))
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        out = sampler(logits)
    assert tuple(out.shape) == (5,)
    names = [e.name for e in prof.events() if e.name.startswith("MODULE: Sampler")]
    assert names and "NUM_REQS(5)" in names[0], names


def test_engine_callables_skip_modules():
    from doolyprof.tracer.hooks import engine_callables

    class PlainSampler:
        def __call__(self, logits, input_batch):
            return logits

    class V2Runner:
        sampler = PlainSampler()

    class V1Runner:
        sampler = torch.nn.Identity()  # V1 runner: Sampler is an nn.Module

    assert list(engine_callables(V2Runner())) == ["PlainSampler"]
    assert engine_callables(V1Runner()) == {}
    assert engine_callables(object()) == {}


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except AssertionError as e:
                failures += 1; print(f"FAIL {name}: {e}")
            except Exception as e:  # noqa: BLE001
                failures += 1; print(f"ERROR {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failures else 0)
