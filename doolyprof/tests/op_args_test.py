"""Unit tests for building a raw op's call from its traced inputs
(profiler.prepare_op_args and the layer-name helpers).

* A custom op that names the registered layer it acts on (vLLM 0.33
  North-Mini's ``vllm::cohere2_mlp_join(attn, mlp, layer_name)``) gets the
  innermost registered module around it in the trace: Kineto keeps no value
  for that Python-object argument, so the resolver used to call the op with
  two of its three arguments and drop it from the profile.
* The overloads of a Python-registered op are found even though ``dir()`` does
  not list them before first access.

Runs directly: ``python doolyprof/tests/op_args_test.py``. CPU only.
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import torch  # noqa: E402

from doolyprof.profiler.profiler import (  # noqa: E402
    enclosing_registered_layer, engine_batch_limits, needs_layer_name, prepare_op_args,
    step_token_ratio, unrecordable_required_args)


@torch.library.custom_op("dooly_test::join", mutates_args=())
def _join(attn_output: torch.Tensor, mlp_output: torch.Tensor, layer_name: str) -> torch.Tensor:
    return attn_output + mlp_output


@torch.library.custom_op("dooly_test::scale", mutates_args=())
def _scale(x: torch.Tensor, mode: str = "plain") -> torch.Tensor:
    return x * 2


@torch.library.custom_op("dooly_test::norm", mutates_args=())
def _norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    return x


@torch.library.custom_op("dooly_test::cache_write", mutates_args=())
def _cache_write(x: torch.Tensor, kv_cache_dtype: str, layer_name: str) -> torch.Tensor:
    return x


@torch.library.custom_op("dooly_test::attn_out", mutates_args=("output",))
def _attn_out(query: torch.Tensor, output: torch.Tensor, layer_name: str,
              output_scale: torch.Tensor | None = None) -> None:
    output.copy_(query)


JOIN = torch.ops.dooly_test.join
SCALE = torch.ops.dooly_test.scale
NORM = torch.ops.dooly_test.norm
CACHE_WRITE = torch.ops.dooly_test.cache_write
ATTN_OUT = torch.ops.dooly_test.attn_out


def test_layer_name_traced_as_empty_input_is_missing():
    # vLLM's unified_attention_with_output(q, k, v, output, layer_name, Tensor?
    # output_scale=None, ...): inputs follow the layer name, so its slot is
    # filled -- with Kineto's empty value, None once the input generator has
    # seen the zero-size shape. It read as a value: no binding, and the op ran
    # without a forward context (vLLM's Transformers backend, SmolLM3)
    from doolyprof.profiler.profiler import op_takes_layer_name
    q, out = torch.ones(3, 4), torch.zeros(3, 4)
    inputs = [q, out, None, None]
    assert needs_layer_name(ATTN_OUT, "dooly_test::attn_out", inputs)
    assert unrecordable_required_args(ATTN_OUT, "dooly_test::attn_out", inputs) == ["layer_name"]
    assert unrecordable_required_args(ATTN_OUT, "dooly_test::attn_out", inputs, allow_layer_name=True) == []
    args, kwargs = prepare_op_args(ATTN_OUT, "dooly_test::attn_out", inputs, layer_name="model.layers.1.self_attn.attn")
    assert args[2] == "model.layers.1.self_attn.attn" and args[3] is None, args
    ATTN_OUT(*args, **kwargs)
    assert torch.equal(out, q)
    # a recorded name stays as it is; without a name to bind, the slot is left alone
    assert not needs_layer_name(ATTN_OUT, "dooly_test::attn_out", [q, out, "model.layers.0.attn"])
    args, _ = prepare_op_args(ATTN_OUT, "dooly_test::attn_out", inputs)
    assert args[2] is None
    assert op_takes_layer_name(ATTN_OUT, "dooly_test::attn_out") and not op_takes_layer_name(SCALE, "dooly_test::scale")


def test_none_tensor_argument_does_not_shift_later_inputs():
    # _C::rms_norm(input, weight=None, epsilon) and vLLM's moe_forward_shared(..,
    # input_ids=None, ..): the trace keeps nothing for the None tensor, so the
    # float / int after it landed on the Tensor? slot and the call raised.
    x = torch.zeros(3, 4)
    args, kwargs = prepare_op_args(NORM, "dooly_test::norm", [x, 1e-6])
    assert args[0] is x and args[1] is None and args[2] == 1e-6 and not kwargs, (args, kwargs)
    w = torch.ones(4)
    args, _ = prepare_op_args(NORM, "dooly_test::norm", [x, w, 1e-6])
    assert args[1] is w and args[2] == 1e-6


def test_defaulted_string_not_recorded_keeps_its_default():
    # aten::pad(self, pad, str mode="constant", float? value): the trace keeps
    # no value for mode, so the float value landed on it
    pad = torch.ops.aten.pad
    x = torch.zeros(3, 4)
    args, _ = prepare_op_args(pad, "aten::pad", [x, [0, 2], 1.0])
    assert args[0] is x and args[1] == [0, 2] and args[2] == "constant" and args[3] == 1.0, args
    out = pad(*args)
    assert tuple(out.shape) == (3, 6) and float(out[0, -1]) == 1.0


def test_unrecordable_required_args():
    x = torch.zeros(3, 4)
    # a cache dtype string and a layer name: nothing in the trace fills them
    assert unrecordable_required_args(CACHE_WRITE, "dooly_test::cache_write", [x]) == ["kv_cache_dtype", "layer_name"]
    # a layer name alone can be bound on the last-resort attempt
    assert unrecordable_required_args(JOIN, "dooly_test::join", [x, x]) == ["layer_name"]
    assert unrecordable_required_args(JOIN, "dooly_test::join", [x, x], allow_layer_name=True) == []
    # a layer name the trace kept as an empty string is not a value
    assert unrecordable_required_args(JOIN, "dooly_test::join", [x, x, ""]) == ["layer_name"]
    # a defaulted string and a None tensor slot are not missing
    assert unrecordable_required_args(SCALE, "dooly_test::scale", [x]) == []
    assert unrecordable_required_args(NORM, "dooly_test::norm", [x, 1e-6]) == []


def _event(path=None, parent=None):
    return SimpleNamespace(args={"module_path": path} if path else {}, parent=parent)


def test_needs_layer_name_only_for_unrecorded_required_name():
    t = torch.zeros(3, 4)
    assert needs_layer_name(JOIN, "dooly_test::join", [t, t])
    assert not needs_layer_name(JOIN, "dooly_test::join", [t, t, "model.layers.0"])
    assert not needs_layer_name(SCALE, "dooly_test::scale", [t])          # has a default
    assert not needs_layer_name(torch.ops.aten.add, "aten::add.Tensor", [t, t])


def test_prepare_op_args_passes_layer_name():
    t = torch.ones(3, 4)
    args, kwargs = prepare_op_args(JOIN, "dooly_test::join", [t, t], layer_name="model.layers.3")
    assert len(args) == 3 and args[2] == "model.layers.3" and not kwargs, (args, kwargs)
    assert torch.equal(JOIN(*args), 2 * t)
    # without a layer name the call is left as before
    args, _ = prepare_op_args(JOIN, "dooly_test::join", [t, t])
    assert len(args) == 2


def test_enclosing_registered_layer_innermost_registered_path():
    layer = _event("model.layers.3", _event("model", _event()))
    api = _event(None, _event("model.layers.3.mlp_wrapper", layer))   # an unregistered module in between
    module = SimpleNamespace(hierarchy=SimpleNamespace(anchor=SimpleNamespace(parent=api)))
    registered = {"model.layers.3": object(), "model.layers.3.self_attn.attn": object()}
    assert enclosing_registered_layer(module, registered) == "model.layers.3"
    assert enclosing_registered_layer(module, {"model.layers.9": object()}) is None
    assert enclosing_registered_layer(SimpleNamespace(hierarchy=None), registered) is None


def test_enclosing_registered_layer_called_through_forward():
    # vLLM's Transformers modeling backend (SmolLM3): the HF attention module
    # calls its vLLM Attention child with self_attn.forward(q, k, v), so the
    # attention ops have model.layers.1.self_attn around them and no annotation
    # for model.layers.1.self_attn.attn, the registered layer
    hf_attn = _event("model.layers.1.self_attn", _event("model.layers.1", _event("model", _event())))
    module = SimpleNamespace(hierarchy=SimpleNamespace(anchor=SimpleNamespace(parent=_event(None, hf_attn))))
    registered = {f"model.layers.{i}.self_attn.attn": object() for i in range(4)}
    assert enclosing_registered_layer(module, registered) == "model.layers.1.self_attn.attn"
    # two registered layers under it: which one the op acts on is unknown
    two = dict(registered, **{"model.layers.1.self_attn.attn2": object()})
    assert enclosing_registered_layer(module, two) is None
    # an enclosing registered layer still wins
    assert enclosing_registered_layer(module, dict(registered, **{"model.layers.1": object()})) == "model.layers.1"


def test_absorbing_takes_calls_inside_the_parent_off_each_group():
    # SmolLM3 (vLLM's Transformers backend): the 9 NoPE layers' attention ops
    # resolve to the whole SmolLM3Attention, the 27 RoPE layers' run on their
    # own; the q/k/v linear, one group over all layers, kept every call, so the
    # NoPE layers' were counted twice. Here 4 layers, layer 3 the NoPE one.
    from doolyprof.profiler.resolver import ModuleResolver

    def ev(name, parent):
        return SimpleNamespace(name=name, parent=parent)
    model = ev("MODULE: Model IN:[]", None)
    attn = [ev(f"MODULE: SmolLM3Attention IN:[] PATH:model.layers.{i}.self_attn", ev(f"MODULE: Layer PATH:model.layers.{i}", model))
            for i in range(4)]
    qkv = [ev("API: linear", a) for a in attn]                    # one call per layer
    nope_attn_op = ev("vllm::unified_attention_with_output", attn[3])
    nope_kv = ev("vllm::unified_kv_cache_update", attn[3])

    def group(name, anchors):
        return SimpleNamespace(operation_name=name, hierarchy=SimpleNamespace(anchor=anchors[0]),
                               instance_anchors=list(anchors), count=len(anchors))
    linear, kv, current = group("aten::linear", qkv), group("vllm::unified_kv_cache_update", [nope_kv]), \
        group("vllm::unified_attention_with_output", [nope_attn_op])
    modules = [current, linear, kv]
    resolver = ModuleResolver.__new__(ModuleResolver)
    absorbed_idx, resolved_idx = set(), set()
    names, _ = resolver._absorb_siblings(attn[3], 0, modules, absorbed_idx, resolved_idx, op_anchor=nope_attn_op)
    assert names == ["vllm::unified_kv_cache_update"] and absorbed_idx == {2}, (names, absorbed_idx)
    assert linear.count == 3 and 1 not in absorbed_idx, linear.count   # 3 RoPE layers' calls remain
    # a parent standing for several members covers each member's instance
    rope_ops = [ev("vllm::unified_attention_with_output", a) for a in attn[:3]]
    rope = group("vllm::unified_attention_with_output", rope_ops)
    names, _ = resolver._absorb_siblings(attn[0], 0, [rope, linear], set(), set(), op_anchor=rope_ops[0])
    assert names == ["aten::linear"], names                               # all of its calls are now inside one


def _hierarchy(root_params, anchor_params):
    return SimpleNamespace(root_module=SimpleNamespace(params=root_params),
                           anchor=SimpleNamespace(params=anchor_params))


def test_step_token_ratio_sequence_parallel_share():
    # vLLM 0.33 Qwen3.6 at fake TP2 x DP2 + EP4: the step runs 60 tokens, each
    # TP rank's MoERunner 30; the resolver gave the forward context 30 tokens,
    # vLLM split them into ceil(30 / 2) = 15 per rank and dispatch asserted.
    moe = SimpleNamespace(hierarchy=_hierarchy(
        "[[NUM_TOKS(60)] | [MODEL_CONFIG(3), NUM_TOKS(60)]]",
        "[[NUM_TOKS(30)[hist:NUM_TOKS:30], MODEL_CONFIG(2048)] | [NUM_TOKS(30), MODEL_CONFIG(2048)]]"))
    assert step_token_ratio(moe) == 2
    # Kimi-K3 at TP8 x DP16: 470 tokens, ceil(470 / 8) = 59 per rank
    kimi = SimpleNamespace(hierarchy=_hierarchy("[[NUM_TOKS(470)]]", "[[NUM_TOKS(59), MODEL_CONFIG(7168)]]"))
    assert step_token_ratio(kimi) == 8


def test_step_token_ratio_whole_batch_and_unknown():
    assert step_token_ratio(SimpleNamespace(hierarchy=_hierarchy(
        "[[NUM_TOKS(60)]]", "[[NUM_TOKS(60), MODEL_CONFIG(2048)]]"))) == 1
    # padded / buffer dims are not shares of the batch
    assert step_token_ratio(SimpleNamespace(hierarchy=_hierarchy(
        "[[NUM_TOKS(470)]]", "[[NUM_TOKS(472), MODEL_CONFIG(7168)]]"))) == 1
    assert step_token_ratio(SimpleNamespace(hierarchy=_hierarchy(
        "[[NUM_TOKS(60)]]", "[[MAX_NUM_TOKS(521), MODEL_CONFIG(64)]]"))) == 1
    assert step_token_ratio(SimpleNamespace(hierarchy=_hierarchy(
        "[[NUM_TOKS(60)]]", "[[NUM_REQS(5), MODEL_CONFIG(2048)]]"))) == 1
    assert step_token_ratio(SimpleNamespace(hierarchy=None)) == 1
    assert step_token_ratio(SimpleNamespace()) == 1


def test_engine_batch_limits_follow_trace_and_sweep():
    # Kimi-K3 at TP8 x DP16: vLLM's default 8192-token limit made the MoE
    # workspace 38 GiB at start-up; the trace ran with 521 / 263
    reg = {"521": "MAX_NUM_TOKS", "263": "MAX_NUM_REQS", "60": "NUM_TOKS", "4": "MODEL_CONFIG"}
    assert engine_batch_limits(reg, 2, 128) == (521, 263)
    assert engine_batch_limits(reg, 300, 4096) == (4096, 300)  # sweep needs more
    assert engine_batch_limits({}, 2, 128) == (128, 2)         # trace without the limits
    assert engine_batch_limits(None, 512, 128) == (512, 512)  # tokens >= seqs (vLLM requires it)


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
