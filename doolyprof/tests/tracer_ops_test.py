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


def test_chunk_then_view_keeps_token_dim():
    # vLLM 0.17.1 Qwen3.5 attention: q_gate [T, 2*heads*head_dim] is chunked into
    # q and gate, then q.view(-1, heads, head_dim). The chunk kept the merge
    # history of the uncut dim (MODEL_CONFIG 6144 on a 3072-wide q), so the view
    # saw a phantom factor 2 next to NUM_TOKS and raised.
    from doolyprof.tracer.types import DimTaint, TaintedInt
    from doolyprof.tracer.registry import register_taint
    T, H, D = 60, 12, 256
    mc, nt = Taint("MODEL_CONFIG"), Taint("NUM_TOKS")
    register_taint(T, nt)
    q_gate = TaintedTensor(torch.zeros(T, 2 * H * D),
                           (DimTaint(nt), DimTaint(mc, merge_history={mc: 2 * H * D})))
    q, gate = torch.chunk(q_gate, 2, dim=-1)
    assert q._dim_taints[1].merge_history == {mc: H * D}, q._dim_taints[1]
    out = q.view(-1, TaintedInt(H, "MODEL_CONFIG"), TaintedInt(D, "MODEL_CONFIG"))
    assert tuple(out.shape) == (T, H, D)
    assert _labels(out) == ["NUM_TOKS", "MODEL_CONFIG", "MODEL_CONFIG"], _labels(out)


def test_split_cuts_merge_history():
    from doolyprof.tracer.types import DimTaint
    mc = Taint("MODEL_CONFIG")
    x = TaintedTensor(torch.zeros(4, 96), (DimTaint(Taint("NUM_TOKS")), DimTaint(mc, merge_history={mc: 96})))
    a, b = torch.split(x, 48, dim=1)
    assert a._dim_taints[1].merge_history == {mc: 48} and b._dim_taints[1].merge_history == {mc: 48}


def test_reshape_minus_one_with_nested_dimtaint_arg():
    # vLLM 0.33 Qwen3.6 GDN built for a TP x DP rank: z.reshape(z.size(0), -1,
    # self.head_v_dim) with head_v_dim carrying a DimTaint nested in a DimTaint
    # raised "unhashable type: 'DimTaint'".
    from doolyprof.tracer.types import DimTaint, TaintedInt
    mc = Taint("MODEL_CONFIG")
    x = TaintedTensor(torch.zeros(4, 6 * 8), (Taint("NUM_TOKS"), mc))
    out = x.reshape(x.size(0), -1, TaintedInt(8, DimTaint(DimTaint(mc))))
    assert tuple(out.shape) == (4, 6, 8), tuple(out.shape)
    assert _labels(out)[0] == "NUM_TOKS", _labels(out)


def test_tensor_new_takes_taints_from_size_arguments():
    # vLLM 0.33 Kimi-K3: hidden_states.new_empty(T, num_attn_res_blocks, H). The
    # 8 slots (cdiv(93 layers, 12)) are a MODEL_CONFIG TaintedInt, but the output
    # was labelled by matching sizes against the tensor's own dims, and no input
    # dim has size 8.
    from doolyprof.tracer.types import TaintedInt
    x = TaintedTensor(torch.zeros(59, 64), (Taint("NUM_TOKS"), Taint("MODEL_CONFIG")))
    slots = TaintedInt(8, "MODEL_CONFIG")
    out = x.new_empty(x.size(0), slots, x.size(1))
    assert tuple(out.shape) == (59, 8, 64)
    assert _labels(out) == ["NUM_TOKS", "MODEL_CONFIG", "MODEL_CONFIG"], _labels(out)
    assert _labels(x.new_zeros((x.size(0), slots))) == ["NUM_TOKS", "MODEL_CONFIG"]
    assert _labels(x.new_full((slots, x.size(1)), 1.0)) == ["MODEL_CONFIG", "MODEL_CONFIG"]
    # an untainted size still falls back to the tensor's own dims, as before
    assert _labels(x.new_empty(59, 64)) == ["NUM_TOKS", "MODEL_CONFIG"], _labels(x.new_empty(59, 64))
    # sizes equal to two input dims with different taints: the arguments decide,
    # and without tainted arguments the ambiguity no longer raises
    y = TaintedTensor(torch.zeros(64, 64), (Taint("NUM_TOKS"), Taint("MODEL_CONFIG")))
    out = y.new_empty(TaintedInt(64, "MODEL_CONFIG"), TaintedInt(64, "MODEL_CONFIG"))
    assert _labels(out) == ["MODEL_CONFIG", "MODEL_CONFIG"], _labels(out)
    assert tuple(y.new_empty(64, 64).shape) == (64, 64)


def test_view_minus_one_keeps_token_taint_equal_to_config_value():
    # vLLM 0.33 Kimi-K3 MoE at TP8 x DP8 (sequence parallel): view(-1, hidden)
    # on 59 tokens per rank. 59 is also a config value (the layer-index lists
    # run up to 93), and the -1 dim took the registry's MODEL_CONFIG label
    # although the only input component left was NUM_TOKS: 59.
    from doolyprof.tracer.registry import register_taint
    from doolyprof.tracer.types import TaintedInt
    register_taint(59, Taint("MODEL_CONFIG"))
    x = TaintedTensor(torch.zeros(59, 64), (Taint("NUM_TOKS"), Taint("MODEL_CONFIG")))
    n, hidden = x.shape
    assert _labels(x.view(-1, hidden)) == ["NUM_TOKS", "MODEL_CONFIG"], _labels(x.view(-1, hidden))
    out = x.reshape(-1, TaintedInt(64, "MODEL_CONFIG"))
    assert _labels(out) == ["NUM_TOKS", "MODEL_CONFIG"], _labels(out)


def test_view_dtype_keyword_keeps_taints():
    # vLLM 0.33 DeepGEMM MoE (picked on Blackwell for Kimi-K3's MXFP4 experts):
    # workspace.view(dtype=torch.float8_e4m3fn) raised "TensorBase.view() got an
    # unexpected keyword argument 'dtype'" inside the tracer's view wrapper
    x = _tokens_by_hidden(torch.bfloat16)
    out = x.view(dtype=torch.float16)
    assert out.dtype == torch.float16 and _labels(out) == ["NUM_TOKS", "MODEL_CONFIG"], _labels(out)
    out = x.view(dtype=torch.int8)
    assert tuple(out.shape) == (9, 128) and _labels(out) == ["NUM_TOKS", None], _labels(out)
    assert tuple(x.view(size=(9, 64)).shape) == (9, 64)
    assert tuple(x.reshape(shape=(64, 9)).shape) == (64, 9)


def test_view_minus_one_with_untainted_component():
    # vLLM 0.33 native FP8 linear on Blackwell (Mixtral FP8): the quantized
    # activation's hidden dim has an untainted component, and
    # x.view(-1, x.shape[-1]) left {NUM_TOKS: 65, None: 4096} -> "registry entry
    # is pure taint but we have multiple remaining components". The plain 4096
    # takes the untainted part, leaving NUM_TOKS for the -1 dim.
    from doolyprof.tracer.registry import register_taint
    from doolyprof.tracer.types import DimTaint
    nt = Taint("NUM_TOKS")
    register_taint(65, nt)
    x = TaintedTensor(torch.zeros(65, 4096), (DimTaint(nt), DimTaint(None, merge_history={None: 4096})))
    out = x.view(-1, x.shape[-1])
    assert tuple(out.shape) == (65, 4096)
    assert _labels(out)[0] == "NUM_TOKS", _labels(out)


def test_untainted_component_used_only_when_sizes_agree():
    # The untainted part is consumed only when the plain sizes divide it and the
    # leftover components multiply to the -1 dim's size; otherwise nothing changes
    from doolyprof.tracer.hooks import _consume_untainted
    nt = Taint("NUM_TOKS")
    rem = {nt: 65, None: 4096}
    assert _consume_untainted(rem, (-1, 4096), 0, 65) == {nt: 65}
    assert _consume_untainted(rem, (-1, 32, 128), 0, 65) == {nt: 65}
    assert _consume_untainted(rem, (-1, 2048), 0, 130) == {nt: 65, None: 2}
    assert _consume_untainted(rem, (65, -1), 1, 4096) is rem      # 4096 % 65 != 0
    assert _consume_untainted(rem, (-1, 1024), 0, 65) is rem      # leftover 65*4 != 65
    assert _consume_untainted({nt: 65}, (-1, 4096), 0, 65) == {nt: 65}


_inner_linear = torch.nn.Linear(8, 4)


@torch.library.custom_op("dooly_test::linear_inside", mutates_args=())
def _linear_inside(x: torch.Tensor) -> torch.Tensor:
    # a module called from a custom op's Python body, like vLLM's MoE runner
    # calling its shared experts inside moe_forward_shared
    return _inner_linear(x).detach()


@_linear_inside.register_fake
def _linear_inside_fake(x):
    return x.new_empty(x.shape[0], 4)


def test_module_inside_custom_op_takes_the_ops_input_taints():
    # The tracer runs a custom op on plain tensors, so a module called inside
    # it annotated every dim ?(n) (vLLM 0.33 MoE shared experts, ShortConv's
    # out_proj, MoE dispatch / combine payloads). A dim now takes the taint of
    # the op's input dim of the same size when exactly one taint has it.
    from torch.profiler import profile, ProfilerActivity
    x = TaintedTensor(torch.zeros(5, 8), (Taint("NUM_TOKS"), Taint("MODEL_CONFIG")))
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        torch.ops.dooly_test.linear_inside(x)
    names = [e.name for e in prof.events() if e.name.startswith("MODULE: Linear")]
    assert names and "IN:[[NUM_TOKS(5), MODEL_CONFIG(8)]]" in names[0], names


class _StrictSizeModule(torch.nn.Module):
    # like FlashInfer's chunk_gated_delta_rule: an exact int check on a size
    def forward(self, q):
        n = q.size(0)
        if type(n) is not int:
            raise ValueError("max_seqlen must be a nonnegative integer")
        return q * 2


_strict_inner = _StrictSizeModule()


@torch.library.custom_op("dooly_test::strict_inside", mutates_args=())
def _strict_inside(x: torch.Tensor) -> torch.Tensor:
    return _strict_inner(x).detach()


@_strict_inside.register_fake
def _strict_inside_fake(x):
    return torch.empty_like(x)


def test_module_inside_custom_op_falls_back_to_plain_inputs():
    # vLLM 0.33 Qwen3.6 GDN on Blackwell calls FlashInfer's
    # chunk_gated_delta_rule from a custom op's body; with the re-tainted input
    # q.size(0) is a TaintedInt and FlashInfer's `type(...) is not int` raised.
    # The module is called again with the op's plain tensors; its annotation
    # keeps the re-tainted dims.
    from torch.profiler import profile, ProfilerActivity
    x = TaintedTensor(torch.ones(5, 8), (Taint("NUM_TOKS"), Taint("MODEL_CONFIG")))
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        out = torch.ops.dooly_test.strict_inside(x)
    assert torch.equal(torch.Tensor.as_subclass(out, torch.Tensor), torch.full((5, 8), 2.0))
    names = [e.name for e in prof.events() if e.name.startswith("MODULE: _StrictSizeModule")]
    assert names and "IN:[[NUM_TOKS(5), MODEL_CONFIG(8)]]" in names[0], names


@torch.library.custom_op("dooly_test::strict_after_linear", mutates_args=())
def _strict_after_linear(x: torch.Tensor) -> torch.Tensor:
    # as in Qwen3.6's GDN op body: a projection's output (tainted, from the
    # re-tainted input) is what reaches the strict library call
    return _strict_inner(_inner_linear(x)).detach()


@_strict_after_linear.register_fake
def _strict_after_linear_fake(x):
    return x.new_empty(x.shape[0], 4)


def test_module_inside_custom_op_strips_taints_of_inner_outputs():
    from torch.profiler import profile, ProfilerActivity
    x = TaintedTensor(torch.ones(5, 8), (Taint("NUM_TOKS"), Taint("MODEL_CONFIG")))
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        out = torch.ops.dooly_test.strict_after_linear(x)
    ref = _inner_linear(torch.ones(5, 8)).detach() * 2
    got = torch.Tensor.as_subclass(out, torch.Tensor)
    assert torch.allclose(got, ref), ("values", got.tolist(), ref.tolist())
    names = [e.name for e in prof.events() if e.name.startswith("MODULE: _StrictSizeModule")]
    assert names and "NUM_TOKS(5)" in names[0], ("annotation", names)


def test_tainted_int_product_with_nested_dimtaint():
    # vLLM 0.33 FlashInfer backend on Blackwell decodes with FlashInfer's XQA
    # kernel, which computes num_pages_per_seq * page_size from the tainted
    # attention metadata; a DimTaint nested in a DimTaint became a dict key
    # ("unhashable type: 'DimTaint'")
    from doolyprof.tracer.types import DimTaint, TaintedInt
    pages = TaintedInt(7, DimTaint(DimTaint(Taint("NUM_REQS"))))
    page_size = TaintedInt(16, Taint("MODEL_CONFIG"))
    for prod in (pages * page_size, page_size * pages):
        assert int(prod) == 112
        hist = prod.taint.merge_history
        assert {str(k) for k in hist} == {"NUM_REQS", "MODEL_CONFIG"}, hist
    same = TaintedInt(3, DimTaint(DimTaint(Taint("NUM_REQS")))) * TaintedInt(5, Taint("NUM_REQS"))
    assert int(same) == 15 and str(same.taint.taint) == "NUM_REQS", same.taint


def test_buffer_sized_from_a_weight_shape_keeps_its_label():
    # vLLM 0.33 CUTLASS FP8 GEMM (Blackwell): out = torch.empty((a.shape[0],
    # b.shape[1])) with b the weight Parameter; Parameter.shape gave plain ints
    # and Mixtral-8x7B's fused QKV output was ?(6144) in every layer
    from doolyprof.tracer.hooks import parameter_size
    from doolyprof.tracer.types import DimTaint
    mc = Taint("MODEL_CONFIG")
    w = torch.nn.Parameter(torch.zeros(16, 48), requires_grad=False)
    object.__setattr__(w, "_dim_taints", (DimTaint(mc), DimTaint(mc)))
    a = TaintedTensor(torch.zeros(5, 16), (Taint("NUM_TOKS"), mc))
    saved = torch.nn.Parameter.__dict__.get("size"), torch.nn.Parameter.__dict__.get("shape")
    torch.nn.Parameter.size, torch.nn.Parameter.shape = parameter_size, property(parameter_size)
    try:
        out = torch.empty((a.shape[0], w.shape[1]))
        assert _labels(out) == ["NUM_TOKS", "MODEL_CONFIG"], _labels(out)
        assert _labels(torch.empty(w.size(1))) == ["MODEL_CONFIG"]
        from doolyprof.tracer import disable_taint_tracking
        disable_taint_tracking()                                    # init / warm-up: plain sizes
        try:
            from doolyprof.tracer.types import TaintedShape
            assert not isinstance(w.shape, TaintedShape) and type(w.size(0)) is int, (type(w.shape), type(w.size(0)))
        finally:
            enable_taint_tracking()
    finally:
        for name, val in zip(("size", "shape"), saved):
            if val is None:
                delattr(torch.nn.Parameter, name)
            else:
                setattr(torch.nn.Parameter, name, val)


def test_view_plain_size_takes_the_matching_input_component():
    # vLLM 0.33 FP8 linear on Blackwell: the CUTLASS output [NUM_TOKS 65,
    # MODEL_CONFIG 6144] is reshaped with view(num_tokens, 6144), the 6144 a
    # plain int; the fused QKV dim lost its label before the q/k/v split
    from doolyprof.tracer.types import DimTaint, TaintedInt
    nt, mc = Taint("NUM_TOKS"), Taint("MODEL_CONFIG")
    x = TaintedTensor(torch.zeros(65, 6144), (DimTaint(nt, merge_history={nt: 65}),
                                              DimTaint(mc, merge_history={mc: 6144})))
    assert _labels(x.view(TaintedInt(65, "NUM_TOKS"), 6144)) == ["NUM_TOKS", "MODEL_CONFIG"]
    assert _labels(x.view(65, 6144)) == ["NUM_TOKS", "MODEL_CONFIG"]
    out = x.view(TaintedInt(65, "NUM_TOKS"), 48, 128)       # a plain size matching no component
    unlabelled = lambda t: [d is None or d.taint is None for d in getattr(t, "_dim_taints", ())]  # noqa: E731
    assert _labels(out)[0] == "NUM_TOKS" and unlabelled(out)[1:] == [True, True], _labels(out)
    sq = TaintedTensor(torch.zeros(64, 64), (DimTaint(nt), DimTaint(mc)))   # 64 is both: ambiguous
    out = sq.view(64, 64)
    assert not isinstance(out, TaintedTensor) or all(unlabelled(out)), _labels(out)


def test_custom_op_int_from_fixed_sizes_is_fixed():
    # vLLM 0.33 Kimi-K3 KDA on Blackwell: workspace_size =
    # torch.ops._flashkda_C.get_workspace_size(max_num_batched_tokens, heads,
    # max_num_seqs) came back a plain int, and the workspace was ?(49102848)
    from doolyprof.tracer.hooks import fixed_int_result
    from doolyprof.tracer.types import TaintedInt
    fixed = (TaintedInt(521, "MAX_NUM_TOKS"), TaintedInt(12, "MODEL_CONFIG"), TaintedInt(263, "MAX_NUM_REQS"))
    out = fixed_int_result(49102848, fixed, {})
    assert isinstance(out, TaintedInt) and int(out) == 49102848 and str(out.taint) == "MODEL_CONFIG", out
    # registered, so a plain buffer of that size (a view of vLLM's persistent
    # workspace) reads as fixed in annotations
    from doolyprof.tracer.tensor import _format_dim
    assert _format_dim(49102848, None).startswith("MODEL_CONFIG(49102848)"), _format_dim(49102848, None)
    assert type(fixed_int_result(7, (TaintedInt(65, "NUM_TOKS"), 4), {})) is int     # follows the batch
    assert type(fixed_int_result(7, (3, 4), {})) is int                             # constants only
    assert type(fixed_int_result(7, (TaintedInt(12, "MODEL_CONFIG"), torch.zeros(2)), {})) is int  # tensor input
    assert type(fixed_int_result(1, fixed, {})) is int
    lib = torch.library.Library("dooly_test_int", "DEF")
    lib.define("ws(int a, int b) -> int")
    lib.impl("ws", lambda a, b: a * b * 3, "CompositeExplicitAutograd")
    got = torch.ops.dooly_test_int.ws(TaintedInt(521, "MAX_NUM_TOKS"), TaintedInt(12, "MODEL_CONFIG"))
    assert isinstance(got, TaintedInt) and int(got) == 521 * 12 * 3, got
    assert type(torch.ops.dooly_test_int.ws(TaintedInt(65, "NUM_TOKS"), 2)) is int


def test_blackbox_size_dividing_config_dims_is_config():
    # vLLM 0.33 Kimi-K3 MoE on Blackwell sends FP8 block scales [tokens, 3584 / 128]
    # through its dispatch inside the MoE op: 28 matches no input dim of the op
    from doolyprof.tracer.tensor import _BLACKBOX_INPUTS, _dim_taint_from_scope
    nt, mc = Taint("NUM_TOKS"), Taint("MODEL_CONFIG")
    _BLACKBOX_INPUTS.append([((59, 3584), (nt, mc)), ((59, 16), (nt, mc))])
    try:
        assert str(_dim_taint_from_scope(28).taint) == "MODEL_CONFIG"
        assert str(_dim_taint_from_scope(59).taint) == "NUM_TOKS"
        assert _dim_taint_from_scope(30) is None                 # divides nothing
    finally:
        _BLACKBOX_INPUTS.pop()
    _BLACKBOX_INPUTS.append([((64, 3584), (nt, mc))])            # a batch dim is a multiple too
    try:
        assert _dim_taint_from_scope(32) is None
    finally:
        _BLACKBOX_INPUTS.pop()


def test_tainted_int_with_numpy_array_operand():
    # vLLM 0.33 chunked local attention (Llama-4): attn_chunk_size -
    # ((seq_lens_np - q_seqlens) % attn_chunk_size), attn_chunk_size a config
    # TaintedInt; the array result was looked up in the registry ("unhashable
    # type: 'numpy.ndarray'")
    import numpy as np
    from doolyprof.tracer.types import TaintedInt, TaintedFloat
    chunk = TaintedInt(8192, "MODEL_CONFIG")
    arr = np.array([5, 9000, 17000])
    for out, ref in ((chunk - arr % chunk, 8192 - arr % 8192), (chunk + arr, 8192 + arr),
                     (chunk * arr, 8192 * arr), (arr - chunk, arr - 8192),
                     (TaintedFloat(0.5, "MODEL_CONFIG") * arr, 0.5 * arr)):
        assert isinstance(out, np.ndarray) and (out == ref).all(), (type(out), out)
    s = chunk - 192                              # scalars keep their taint as before
    assert isinstance(s, TaintedInt) and int(s) == 8000


def test_ellipsis_index_with_new_axes_keeps_labels_in_place():
    # Hugging Face's rotary embedding (SmolLM3 through vLLM's Transformers
    # backend): position_ids[..., None] counted the None as an indexed dim, kept
    # one dim under the Ellipsis and moved NUM_TOKS onto the new size-1 axis,
    # which then read NUM_TOKS(1) in every layer
    from doolyprof.tracer.types import DimTaint
    nt, mc = Taint("NUM_TOKS"), Taint("MODEL_CONFIG")
    pos = TaintedTensor(torch.arange(60).view(1, 60), (None, DimTaint(nt)))
    x = TaintedTensor(torch.zeros(60, 64), (DimTaint(nt), DimTaint(mc)))
    lab = lambda t: [None if d is None or d.taint is None else str(d.taint) for d in t._dim_taints]  # noqa: E731
    assert lab(pos[..., None]) == [None, "NUM_TOKS", None], lab(pos[..., None])
    assert lab(pos[None, ...]) == [None, None, "NUM_TOKS"]
    assert lab(x[..., None, :]) == ["NUM_TOKS", None, "MODEL_CONFIG"], lab(x[..., None, :])
    assert lab(x[None, ..., None]) == [None, "NUM_TOKS", "MODEL_CONFIG", None], lab(x[None, ..., None])
    assert lab(x[..., 0]) == ["NUM_TOKS"]
    assert lab(x[1:, ...]) == ["NUM_TOKS-1", "MODEL_CONFIG"], lab(x[1:, ...])   # the batch-slice offset rule


def test_reshape_inside_blackbox_registers_nothing():
    # vLLM's Marlin MoE on an A100 cuts a fixed buffer to a batch-dependent
    # length (x.flatten()[:n].view(..)) and then calls .view(-1, topk, K) inside
    # the MoE custom op; the -1 (59 tokens x 64 ranks) went into the registry as
    # MODEL_CONFIG and Kimi-K3's EP combine payload read MODEL_CONFIG(3776)
    from doolyprof.tracer.registry import TAINT_REGISTRY
    from doolyprof.tracer.tensor import _BLACKBOX_INPUTS
    from doolyprof.tracer.types import DimTaint, TaintedInt
    mc = Taint("MODEL_CONFIG")
    rows, topk, k = 3779, 4, 8                       # sizes no other test registers
    x = TaintedTensor(torch.zeros(rows * topk, k), (DimTaint(mc, merge_history={mc: rows * topk}), DimTaint(mc)))
    _BLACKBOX_INPUTS.append([((7, 8), (Taint("NUM_TOKS"), mc))])
    try:
        out = x.view(-1, TaintedInt(topk, "MODEL_CONFIG"), TaintedInt(k, "MODEL_CONFIG"))
    finally:
        _BLACKBOX_INPUTS.pop()
    assert tuple(out.shape) == (rows, topk, k)
    assert rows not in TAINT_REGISTRY, TAINT_REGISTRY.get(rows)
    out = x.view(-1, TaintedInt(topk, "MODEL_CONFIG"), TaintedInt(k, "MODEL_CONFIG"))   # outside: as before
    assert rows in TAINT_REGISTRY


def test_module_tainted_tensors_get_their_unlabelled_dims_filled():
    # Gemma-3's global rotary cache is a TaintedTensor built at init from a
    # labelled head size and a plain max_position x scaling-factor row count;
    # label_module_tensors skipped TaintedTensors, so its rows stayed ?(1048576)
    from doolyprof.tracer.registry import label_module_tensors
    from doolyprof.tracer.types import DimTaint
    rope = torch.nn.Module()
    rope.cos_sin_cache = TaintedTensor(torch.zeros(1777, 16), (None, DimTaint(Taint("MODEL_CONFIG"))))
    reg = {16: Taint("MODEL_CONFIG")}
    assert label_module_tensors(rope, registry=reg) == 1
    assert [str(d.taint) for d in rope.cos_sin_cache._dim_taints] == ["MODEL_CONFIG", "MODEL_CONFIG"]
    assert str(reg[1777]) == "MODEL_CONFIG"


def test_strip_taints_keeps_storage_and_values():
    from doolyprof.tracer.tensor import strip_taints
    from doolyprof.tracer.types import TaintedInt
    t = TaintedTensor(torch.zeros(3, 4), (Taint("NUM_TOKS"), None))
    args, kwargs, changed = strip_taints((t, [TaintedInt(3, "NUM_TOKS"), 2]), {"n": TaintedInt(4, "MODEL_CONFIG")})
    assert changed and not isinstance(args[0], TaintedTensor) and type(args[1][0]) is int and type(kwargs["n"]) is int
    args[0][0, 0] = 7.0
    assert float(torch.Tensor.as_subclass(t, torch.Tensor)[0, 0]) == 7.0
    assert strip_taints((torch.zeros(2), 3), {})[2] is False


def test_unflatten_takes_taints_from_its_sizes():
    # vLLM 0.33 Gemma 4: k.unflatten(-1, (num_kv_heads, head_dim)) on [T, 1024]
    # left the 4 KV heads per rank untracked (no input dim has size 4).
    from doolyprof.tracer.types import TaintedInt
    k = TaintedTensor(torch.zeros(60, 1024), (Taint("NUM_TOKS"), Taint("MODEL_CONFIG")))
    heads, head_dim = TaintedInt(4, "MODEL_CONFIG"), TaintedInt(256, "MODEL_CONFIG")
    out = k.unflatten(-1, (heads, head_dim))
    assert tuple(out.shape) == (60, 4, 256), tuple(out.shape)
    assert _labels(out) == ["NUM_TOKS", "MODEL_CONFIG", "MODEL_CONFIG"], _labels(out)
    out = torch.unflatten(k, 1, (heads, -1))
    assert tuple(out.shape) == (60, 4, 256), tuple(out.shape)
    assert _labels(out) == ["NUM_TOKS", "MODEL_CONFIG", "MODEL_CONFIG"], _labels(out)


class _Expert(torch.nn.Module):
    """A shared expert's shape: gate_up_proj to 2 x intermediate, then an
    activation module with no weights of its own."""
    def __init__(self):
        super().__init__()
        self.gate_up_proj = torch.nn.Linear(8, 12, bias=False)
        self.act_fn = torch.nn.SiLU()

    def forward(self, x):
        return self.act_fn(self.gate_up_proj(x))


_inner_expert = _Expert()
# vLLM builds its weights from the tainted config, so they carry dim taints
# (Kimi-K3's dense gate_up_proj weight: [MODEL_CONFIG(67584), MODEL_CONFIG(7168)])
from doolyprof.tracer.types import DimTaint as _DimTaint  # noqa: E402
_inner_expert.gate_up_proj.weight._dim_taints = (
    _DimTaint.from_taint(Taint("MODEL_CONFIG")), _DimTaint.from_taint(Taint("MODEL_CONFIG")))


@torch.library.custom_op("dooly_test::expert_inside", mutates_args=())
def _expert_inside(x: torch.Tensor) -> torch.Tensor:
    return _inner_expert(x).detach().clone()


@_expert_inside.register_fake
def _expert_inside_fake(x):
    return x.new_empty(x.shape[0], 12)


def test_module_after_a_linear_inside_custom_op_keeps_config_dims():
    # vLLM 0.33 MoE shared experts inside moe_forward_shared: the activation's
    # input (2 x the intermediate size, from gate_up_proj's weight) was
    # ?(12288) on Kimi-K3, ?(11264) on Qwen1.5-MoE. The module's inputs are
    # now re-tainted inside the op, so its linear propagates the weight's dim.
    from torch.profiler import profile, ProfilerActivity
    x = TaintedTensor(torch.zeros(5, 8), (Taint("NUM_TOKS"), Taint("MODEL_CONFIG")))
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        out = torch.ops.dooly_test.expert_inside(x)
    assert tuple(out.shape) == (5, 12)
    names = [e.name for e in prof.events() if e.name.startswith("MODULE: SiLU")]
    assert names and "IN:[[NUM_TOKS(5)" in names[0] and "MODEL_CONFIG(12)" in names[0], names
    # the op's own ops still write no taint annotation of their own
    inner = [e.name for e in prof.events() if e.name.startswith("API: linear")]
    assert not inner, inner


def test_empty_dim_is_labelled_not_untracked():
    from doolyprof.tracer.tensor import _format_dim
    assert _format_dim(0, None) == "EMPTY(0)"
    assert _format_dim(1, None) == "SINGLETON(1)"


@torch.library.custom_op("dooly_test::half_rows", mutates_args=())
def _half_rows(x: torch.Tensor) -> torch.Tensor:
    return x[: -(-x.shape[0] // 2)].clone()


@_half_rows.register_fake
def _half_rows_fake(x):
    # written like vLLM's sequence_parallel_chunk_impl_fake
    shape = list(x.shape)
    shape[0] = -(-x.size(0) // 2)
    return torch.empty(shape, dtype=x.dtype, device=x.device)


def test_custom_op_output_dim_taint_from_fake_impl():
    # vLLM 0.33 sequence-parallel MoE (fake TP2 x DP2): the custom op cuts the
    # 60 tokens to 30 per TP rank on plain tensors; nothing in the inputs is 30,
    # so layer 0's MoE ran on untainted rows. The op's fake implementation,
    # evaluated on the tainted input, gives ceil(T / 2) with the token taint.
    x = TaintedTensor(torch.zeros(6, 4), (Taint("NUM_TOKS"), Taint("MODEL_CONFIG")))
    out = torch.ops.dooly_test.half_rows(x)
    assert tuple(out.shape) == (3, 4), tuple(out.shape)
    assert _labels(out) == ["NUM_TOKS", "MODEL_CONFIG"], _labels(out)


def test_batch_size_plus_constant_is_an_offset_label():
    # cu_seqlens / query_start_loc have num_reqs + 1 entries. Labelled
    # NUM_REQS, Kimi-K3's 94-request step showed NUM_REQS(95) next to
    # NUM_REQS(94), and the profiler sized such tensors to the request count.
    from doolyprof.tracer.types import TaintedInt
    n = TaintedInt(5, "NUM_REQS")

    def lab(v):
        return str(v.taint)
    assert (int(n + 1), lab(n + 1)) == (6, "NUM_REQS+1")
    assert lab(1 + n) == "NUM_REQS+1"
    assert lab(n - 1) == "NUM_REQS-1"
    assert lab((n + 1) - 1) == "NUM_REQS"
    assert lab(TaintedInt(263, "MAX_NUM_REQS") + 1) == "MAX_NUM_REQS+1"
    # any other arithmetic on it gives the batch label, as before offsets
    assert lab((n + 15) // 16) == "NUM_REQS"
    assert lab((n + 1) * 4) == "NUM_REQS"


def test_query_start_loc_buffer_keeps_request_offsets():
    # vLLM 0.33 V2 runner: InputBuffers.query_start_loc = torch.zeros(
    # max_num_reqs + 1) at init, refilled each step by copy_ from a tensor
    # built in numpy, then query_start_loc[:num_reqs + 1]; the logits indices
    # are query_start_loc[1:] - 1.
    import numpy as np
    from doolyprof.tracer.types import TaintedInt
    max_reqs, n = TaintedInt(263, "MAX_NUM_REQS"), TaintedInt(5, "NUM_REQS")
    buf = torch.zeros(max_reqs + 1, dtype=torch.int32)
    assert _labels(buf) == ["MAX_NUM_REQS+1"], _labels(buf)
    buf.copy_(torch.from_numpy(np.zeros(int(max_reqs) + 1, dtype=np.int32)))
    assert _labels(buf) == ["MAX_NUM_REQS+1"], _labels(buf)  # a plain source keeps it
    qsl = buf[: n + 1]
    assert _labels(qsl) == ["NUM_REQS+1"], _labels(qsl)
    assert _labels(qsl[1:]) == ["NUM_REQS"], _labels(qsl[1:])
    assert _labels(qsl[:-1]) == ["NUM_REQS"], _labels(qsl[:-1])
    assert _labels(qsl[1:-1]) == ["NUM_REQS-1"], _labels(qsl[1:-1])
    assert _labels(qsl[-2:]) == [None], _labels(qsl[-2:])  # two entries, not a batch
    # InputBatch.make_dummy: cumsum(seq_lens, out=query_start_loc[1 : num_reqs + 1])
    # at vLLM's profiling run, where num_reqs is the MAX_NUM_REQS bound
    out = buf[1 : max_reqs + 1]
    assert _labels(out) == ["MAX_NUM_REQS"], _labels(out)
    seq_lens = torch.zeros(max_reqs, dtype=torch.int32)
    torch.cumsum(seq_lens, dim=0, out=out)
    assert _labels(buf[1 : n + 1]) == ["NUM_REQS"], _labels(buf[1 : n + 1])


def _query_start_loc(n_reqs=5):
    import numpy as np
    from doolyprof.tracer.types import TaintedInt
    max_reqs, n = TaintedInt(263, "MAX_NUM_REQS"), TaintedInt(n_reqs, "NUM_REQS")
    buf = torch.zeros(max_reqs + 1, dtype=torch.int32)
    buf.copy_(torch.from_numpy(np.arange(int(max_reqs) + 1, dtype=np.int32) * 12))
    return buf[: n + 1]


def test_cat_of_constant_and_batch_part_is_an_offset_label():
    # FLA's prepare_chunk_offsets (vLLM's gated-delta-rule metadata):
    # cat([cu.new_zeros(1), cdiv(cu[1:] - cu[:-1], 64)]).cumsum(-1), one entry
    # per request plus a leading zero
    cu = _query_start_loc()
    assert _labels(cu) == ["NUM_REQS+1"], _labels(cu)
    chunks = (cu[1:] - cu[:-1] + 63) // 64
    assert _labels(chunks) == ["NUM_REQS"], _labels(chunks)
    offsets = torch.cat([cu.new_zeros(1), chunks]).cumsum(-1)
    assert _labels(offsets) == ["NUM_REQS+1"], _labels(offsets)
    assert offsets.shape[0] == 6
    # a constant on either side, and a fixed (MODEL_CONFIG) part, count too
    fixed = TaintedTensor(torch.zeros(3, dtype=torch.int32), (Taint("MODEL_CONFIG"),))
    assert _labels(torch.cat([chunks, fixed])) == ["NUM_REQS+3"], _labels(torch.cat([chunks, fixed]))
    # two batch parts, or a part of unknown size, are not an offset of one batch size
    assert _labels(torch.cat([chunks, chunks])) != ["NUM_REQS+5"]
    unknown = torch.zeros(4, dtype=torch.int32)
    lab = _labels(torch.cat([chunks, unknown]))[0]
    assert lab is None or "+" not in lab, lab


def test_narrow_takes_its_length_label():
    # vLLM's sequence_parallel_chunk_impl: chunk = y.shape[0] // tp_size, then
    # torch.narrow(y, 0, tp_rank * chunk, chunk)
    y = TaintedTensor(torch.zeros(60, 2048, dtype=torch.bfloat16), (Taint("NUM_TOKS"), Taint("MODEL_CONFIG")))
    chunk = y.shape[0] // 2
    out = torch.narrow(y, 0, 0 * chunk, chunk)
    assert tuple(out.shape) == (30, 2048)
    assert _labels(out) == [str(chunk.taint), "MODEL_CONFIG"], (_labels(out), chunk.taint)
    assert _labels(out.clone()) == _labels(out)
    assert _labels(y.narrow(0, 0, chunk)) == _labels(out)
    # a plain length keeps the generic result (no label from the length)
    assert _labels(y.narrow(1, 0, 1024))[0] == "NUM_TOKS"


def test_kv_cache_dims_are_labelled():
    import types
    from doolyprof.tracer.registry import kv_cache_tensors, register_kv_cache_dims
    mla = torch.zeros(396, 768, 6)
    ctx = {
        "layers.0.attn": types.SimpleNamespace(kv_cache=[mla]),                  # one per virtual engine
        "layers.1.attn": types.SimpleNamespace(kv_cache=mla),                    # the same tensor again
        "layers.2.kda": types.SimpleNamespace(kv_cache=(torch.zeros(396, 12, 5), torch.zeros(396, 12, 9))),
        "layers.3.attn": types.SimpleNamespace(kv_cache=[torch.tensor([])]),     # unbound placeholder
        "layers.4.mlp": types.SimpleNamespace(),                                 # no cache at all
    }
    tensors = kv_cache_tensors(ctx)
    assert len(tensors) == 3, [tuple(t.shape) for t in tensors]
    reg = {12: Taint("MODEL_CONFIG"), 6: Taint("MODEL_CONFIG"), 5: Taint("MODEL_CONFIG"), 9: Taint("MODEL_CONFIG")}
    added = register_kv_cache_dims(tensors, registry=reg)
    assert added == {396: "KV_CACHE", 768: "KV_CACHE"}, added
    assert str(reg[12]) == "MODEL_CONFIG" and str(reg[396]) == "KV_CACHE"


def test_device_block_still_places_new_tensors():
    # vLLM builds its model inside `with torch.device("cuda")` and records
    # torch.get_default_device() as each layer's load device; with the
    # tracer's factory wrappers installed first, new tensors (and that
    # recorded device) came out on the CPU. The hooks are installed at the top
    # of this module, before any device block, as in the tracer.
    with torch.device("meta"):
        assert torch.empty(3).device.type == "meta"
        assert torch.zeros(2, 2).device.type == "meta"
        assert torch.get_default_device().type == "meta"  # via torch.tensor([]) for an index-less device
        assert torch.nn.Linear(2, 2).weight.device.type == "meta"


def test_module_tensors_after_loading_get_fixed_labels():
    # vLLM's Marlin FP8 path rebuilds weight scales after loading ([1, 6144]
    # for Mixtral's fused QKV): unlabelled, or with the labels of the tensor
    # whose .data it replaced
    from doolyprof.tracer.registry import label_module_tensors
    lin = torch.nn.Linear(16, 8)
    lin.weight.data = torch.zeros(1, 6144)               # replaced after the Parameter got labels
    lin.scale = torch.zeros(6144)                        # a plain tensor attribute
    lin.register_buffer("workspace", torch.zeros(521, 4))  # sized like a MAX_NUM_TOKS buffer
    cache = torch.zeros(396, 12)
    lin.kv_cache = cache
    reg = {521: Taint("MAX_NUM_TOKS"), 4: Taint("MODEL_CONFIG")}
    n = label_module_tensors(lin, skip=[cache], registry=reg)
    lab = lambda t: [None if d is None else str(d.taint) for d in t._dim_taints]  # noqa: E731
    assert lab(lin.weight) == [None, "MODEL_CONFIG"], lab(lin.weight)
    assert lab(lin.scale) == ["MODEL_CONFIG"]
    assert lab(lin.workspace) == [None, "MODEL_CONFIG"], lab(lin.workspace)  # 521 left to the registry
    assert getattr(cache, "_dim_taints", None) is None                      # the KV cache is skipped
    assert n == 4, n  # weight, bias, scale, workspace
    # the sizes it labelled are registered (a plain copy, like the cos_sin_cache
    # vLLM's rotary embedding re-creates with .to() at the first forward, still
    # reads as fixed); sizes the registry already knew keep their label
    assert str(reg[6144]) == "MODEL_CONFIG" and str(reg[521]) == "MAX_NUM_TOKS", reg
    assert 396 not in reg and 12 not in reg                                  # nothing from the skipped cache


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
