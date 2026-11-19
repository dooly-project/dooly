"""torch.compile'd functions must run under the tracer's hooks.

vLLM compiles some functions even with enforce_eager (MoE routers such as
grouped_topk, Cohere2-MoE's token_choice_with_bias). They are called from inside
the MoE custom op, where tensors are already plain but scalars read from the
tainted config (top-k, number of expert groups) are still TaintedInt. Dynamo
traced them with the tracer's PatchedSize installed as torch.Size, so any
``x.shape`` in the function failed ("expected namedtuple or structseq but got
<class 'torch.Size'>") and GLM-4.7-Flash, LFM2.5-MoE, North-Mini (Cohere2-MoE)
and Hy-MT2 could not be traced at all.

Runs directly: ``python doolyprof/tests/compile_wrapper_test.py``. CPU only.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import torch  # noqa: E402

from doolyprof.tracer.hooks import install_hooks, PatchedSize  # noqa: E402
from doolyprof.tracer import enable_taint_tracking  # noqa: E402
from doolyprof.tracer.types import TaintedInt, TaintedFloat  # noqa: E402
from doolyprof.tracer.tensor import TaintedTensor  # noqa: E402

install_hooks()
enable_taint_tracking()


# Same shape of code as vLLM's routers: reads x.shape, branches on scalar args.
@torch.compile(dynamic=True, backend="eager")
def route(hidden, gating, topk: int, num_group: int = 0, topk_group: int = 0, scale: float = 1.0):
    if gating.shape[0] == 0:
        return gating, gating
    assert hidden.shape[0] == gating.shape[0]
    if num_group <= 32 and topk <= 32:
        scores = gating.sigmoid()
    else:
        scores = gating.softmax(dim=-1)
    n = scores.size(0)
    groups = scores.view(n, num_group, -1).max(dim=-1).values
    torch.topk(groups, k=topk_group, dim=-1)
    w, i = torch.topk(scores, k=topk, dim=-1)
    return w * scale, i


def _cfg(v):
    return TaintedInt(v, "MODEL_CONFIG")


def test_plain_tensors_tainted_scalars():
    w, i = route(torch.randn(7, 16), torch.randn(7, 64), _cfg(8), _cfg(8), _cfg(4), TaintedFloat(2.5, "MODEL_CONFIG"))
    assert tuple(w.shape) == (7, 8) and tuple(i.shape) == (7, 8)
    assert torch.Size is PatchedSize   # the tracer's patch is back in place afterwards


def test_tainted_tensor_input():
    gating = TaintedTensor(torch.randn(7, 64), (TaintedInt(7, "NUM_TOKS").taint, _cfg(64).taint))
    w, _ = route(torch.randn(7, 16), gating, 8, 8, 4)
    assert isinstance(w, TaintedTensor), type(w)
    assert tuple(torch.Tensor.size(w)) == (7, 8)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
