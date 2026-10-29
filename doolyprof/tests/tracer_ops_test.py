"""Unit tests for tracer taint propagation through tensor ops that vLLM's model
code calls on the hot path.

* ``squeeze(dim=(a, b))`` must accept a tuple of dims (vLLM 0.32 Mamba
  ``bind_kv_cache`` raised ``'<' not supported between tuple and int``).
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
