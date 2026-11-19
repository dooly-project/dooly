"""Unit tests for the two signature fields that identify an op's numeric configuration:

* the module state hash (``doolyprof.tracer.hooks._extract_module_state_hash``), which must
  separate attention layers that differ in KV-cache dtype, logit soft-capping, ALiBi or
  Mamba state dtype while staying identical for ordinary modules, and
* the MoE parallel layout (``FusedMoEParallelConfig`` on the layer's ``moe_config``), which
  must separate TP-sharded experts from expert-parallel ones at the same TP degree, and
* the per-op dtype (``doolyprof.profiler.trace._derive_op_dtype``), derived from the op's own
  traced tensor inputs rather than a run-level constant.

Runs under pytest or directly: ``python doolyprof/tests/signature_fields_test.py``. CPU only.
"""
import enum
import hashlib
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import torch  # noqa: E402

from doolyprof.tracer.hooks import _extract_module_state_hash  # noqa: E402
from doolyprof.profiler.trace import _derive_op_dtype  # noqa: E402
from doolyprof.profiler.finput import ProfileInput  # noqa: E402
from doolyprof.profiler.taint import DimensionType  # noqa: E402

try:
    from vllm.v1.attention.backend import AttentionType
    from vllm.config.cache import CacheConfig
    HAVE_VLLM = True
except Exception:  # pragma: no cover - vLLM optional for the pure-torch tests
    HAVE_VLLM = False


class FakeImpl:
    """Stands in for a vLLM AttentionImpl: the constructor contract is read from __init__."""

    def __init__(self, num_heads, head_size, scale, num_kv_heads, alibi_slopes, sliding_window,
                 kv_cache_dtype, logits_soft_cap, attn_type, kv_sharing_target_layer_name, sinks=None):
        self.num_heads, self.head_size, self.scale = num_heads, head_size, scale
        self.num_kv_heads, self.alibi_slopes, self.sliding_window = num_kv_heads, alibi_slopes, sliding_window
        self.kv_cache_dtype, self.logits_soft_cap, self.attn_type = kv_cache_dtype, logits_soft_cap, attn_type
        self.kv_sharing_target_layer_name, self.sinks = kv_sharing_target_layer_name, sinks
        self.support_trtllm_attn = False  # backend/hardware flag: not a constructor arg, must be ignored


def attention_like(kv_cache_dtype="auto", kv_cache_torch_dtype=torch.bfloat16, sliding_window=None,
                   logits_soft_cap=None, alibi_slopes=None, sinks=None, attn_type="decoder",
                   layer_name="model.layers.0.self_attn.attn", kv_sharing_target_layer_name=None):
    at = AttentionType(attn_type) if HAVE_VLLM else attn_type
    m = types.SimpleNamespace(
        num_heads=32, head_size=128, num_kv_heads=8, sliding_window=sliding_window, has_sink=sinks is not None,
        use_output=True, kv_cache_dtype=kv_cache_dtype, kv_cache_torch_dtype=kv_cache_torch_dtype,
        layer_name=layer_name, kv_sharing_target_layer_name=kv_sharing_target_layer_name, attn_type=at,
        dtype=torch.bfloat16, training=False, quant_config=None, _private=object(),
    )
    m.impl = FakeImpl(32, 128, 0.088, 8, alibi_slopes, sliding_window, kv_cache_dtype, logits_soft_cap, at,
                      kv_sharing_target_layer_name, sinks)
    return m


def test_plain_module_hash_unchanged():
    """Modules without engine config hash exactly as before (primitive attrs only)."""
    m = types.SimpleNamespace(hidden_size=4096, eps=1e-5, training=False, weight_name="w", _p=1)
    canonical = (("eps", 1e-5), ("hidden_size", 4096))
    assert _extract_module_state_hash(m) == hashlib.blake2b(repr(canonical).encode(), digest_size=6).hexdigest()
    assert _extract_module_state_hash(types.SimpleNamespace(name="x")) == ""


def test_kv_cache_dtype_splits_and_auto_canonicalises():
    bf16_auto = _extract_module_state_hash(attention_like("auto", torch.bfloat16))
    bf16_explicit = _extract_module_state_hash(attention_like("bfloat16", torch.bfloat16))
    fp8 = _extract_module_state_hash(attention_like("fp8", torch.uint8))
    fp8_e4m3 = _extract_module_state_hash(attention_like("fp8_e4m3", torch.uint8))
    fp8_e5m2 = _extract_module_state_hash(attention_like("fp8_e5m2", torch.uint8))
    assert bf16_auto != fp8 != fp8_e5m2 and bf16_auto != fp8_e5m2
    if HAVE_VLLM:
        assert bf16_auto == bf16_explicit, "auto and an explicit bfloat16 cache are the same configuration"
        assert fp8 == fp8_e4m3, "bare fp8 is the platform e4m3 flavour"
    # auto resolves through the model dtype, so a float16 model gives a different cache
    assert bf16_auto != _extract_module_state_hash(attention_like("auto", torch.float16))


def test_identifier_strings_do_not_fragment():
    a = _extract_module_state_hash(attention_like(layer_name="model.layers.0.self_attn.attn"))
    b = _extract_module_state_hash(attention_like(layer_name="model.layers.17.self_attn.attn"))
    assert a == b


def test_impl_contract_soft_cap_alibi_sinks():
    base = _extract_module_state_hash(attention_like())
    assert _extract_module_state_hash(attention_like(logits_soft_cap=50.0)) != base
    assert _extract_module_state_hash(attention_like(alibi_slopes=torch.tensor([0.7, 0.5]))) != base
    # tensors contribute shape only: different learned sink values, same hash
    s1 = _extract_module_state_hash(attention_like(sinks=torch.zeros(32)))
    s2 = _extract_module_state_hash(attention_like(sinks=torch.ones(32)))
    assert s1 == s2 and s1 != base
    # a backend/hardware flag on impl that is not a constructor argument is ignored
    m = attention_like(); m.impl.support_trtllm_attn = True
    assert _extract_module_state_hash(m) == base


def test_sliding_window_still_splits():
    assert _extract_module_state_hash(attention_like(sliding_window=None)) != \
        _extract_module_state_hash(attention_like(sliding_window=4096))


def test_enum_attr_by_value():
    if not HAVE_VLLM:
        return
    assert _extract_module_state_hash(attention_like(attn_type="decoder")) != \
        _extract_module_state_hash(attention_like(attn_type="encoder_only"))


def test_mamba_cache_config_dtype():
    if not HAVE_VLLM:
        return
    model_cfg = types.SimpleNamespace(dtype=torch.bfloat16)
    def mixer(cache_dtype):
        return types.SimpleNamespace(num_heads=128, cache_config=CacheConfig(mamba_cache_dtype=cache_dtype,
                                     mamba_ssm_cache_dtype=cache_dtype), model_config=model_cfg, training=False)
    assert _extract_module_state_hash(mixer("auto")) != _extract_module_state_hash(mixer("float32"))
    # the KV cache_dtype field is not a Mamba field: changing it must not split the mixer
    a = mixer("auto"); b = mixer("auto"); b.cache_config.cache_dtype = "fp8"
    assert _extract_module_state_hash(a) == _extract_module_state_hash(b)


def _inp(dtype, shape=(65, 4096), typ="Tensor"):
    return ProfileInput(type=typ, dtype=dtype, dimensions=[], actual_shape=list(shape))


def test_layer_index_attrs_excluded():
    # vLLM stores the layer position on KDA / linear-attention / MoE layers as
    # layer_idx (or layer_id, block_idx); it names the instance, not its config.
    a, b = attention_like(), attention_like()
    a.layer_idx, b.layer_idx = 0, 19
    assert _extract_module_state_hash(a) == _extract_module_state_hash(b)
    a.layer_id, b.layer_id = 3, 4
    a.block_idx, b.block_idx = 1, 2
    assert _extract_module_state_hash(a) == _extract_module_state_hash(b)
    b.sliding_window = 4096  # a real configuration difference must still split
    assert _extract_module_state_hash(a) != _extract_module_state_hash(b)


def test_engine_capacity_attr_excluded():
    """MLA's workspace size follows the engine's max_num_seqs / max_model_len, not the layer."""
    a, b = attention_like(), attention_like()
    a.chunked_prefill_workspace_size, b.chunked_prefill_workspace_size = 504960, 491520
    assert _extract_module_state_hash(a) == _extract_module_state_hash(b)


def test_signature_hash_ignores_descendants():
    """Two anchors that agree on op name, own dims, state hash and dtype hash equal even if the
    ops executed under them differ (one-time autotuning, cached metadata, tracer records)."""
    from doolyprof.profiler.module import ModuleComparator, ModuleInfo
    from doolyprof.profiler.parser import Event, OperationHierarchy

    def ev(name, cat="cpu_op", **args):
        return Event(cat=cat, name=name, ts=0, dur=1, ev_idx=None, external_id=None, python_id=None,
                     python_parent_id=None, args=args)

    def info(children, state="2dcd83b04195", dtype="bfloat16", hidden=2304):
        anchor = ev("KimiDeltaAttention", cat="module")
        for i, c in enumerate(children):
            anchor.children[i] = ev(c)
        h = OperationHierarchy(anchor=anchor, root_module=anchor, nearest_module=anchor, kernels=[], taints=[])
        inp = ProfileInput(type="Tensor", dtype="c10::BFloat16", dimensions=[DimensionType.NUM_TOKENS, DimensionType.MODEL_CONFIG],
                           actual_shape=[504, hidden])
        return ModuleInfo(module_name="KimiDeltaAttention", operation_name="KimiDeltaAttention", inputs=[inp],
                          is_module=True, dtype=dtype, hierarchy=h, state_hash=state)

    cmp = ModuleComparator()
    steady = info(["aten::linear"] * 18 + ["vllm::kda_attention"])
    first = info(["aten::linear"] * 18 + ["vllm::kda_attention", "aten::cumsum", "aten::cat", "PythonSubclass"])
    assert cmp._hash_signature(cmp._module_signature(steady)) == cmp._hash_signature(cmp._module_signature(first))
    assert len(cmp._descendant_signature(first)) == 22 and len(cmp._descendant_signature(steady)) == 19  # still recorded
    for other in (info(["aten::linear"], state="ffffffffffff"), info(["aten::linear"], dtype="float16"),
                  info(["aten::linear"], hidden=4096)):
        assert cmp._hash_signature(cmp._module_signature(steady)) != cmp._hash_signature(cmp._module_signature(other))


def test_impl_class_in_state_hash():
    """Backends sharing one constructor contract must still hash apart."""
    a, b = attention_like(), attention_like()
    class OtherImpl(FakeImpl):
        pass
    b.impl = OtherImpl(32, 128, 0.088, 8, None, None, "auto", None, b.attn_type, None, None)
    assert _extract_module_state_hash(a) != _extract_module_state_hash(b)


def _moe_layer(**layout):
    import dataclasses
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEParallelConfig
    m = torch.nn.Module()
    m.top_k = 8
    cfg = object.__new__(FusedMoEConfig)  # only the fields the state hash reads
    cfg.moe_parallel_config = dataclasses.replace(FusedMoEParallelConfig.make_no_parallel(), **layout)
    cfg.num_local_experts = 64 // layout.get("ep_size", 1)
    m.moe_config = cfg
    return m


def test_moe_parallel_layout_in_state_hash():
    if not HAVE_VLLM:
        print("SKIP (no vLLM)"); return
    tp = _extract_module_state_hash(_moe_layer(tp_size=2))
    ep = _extract_module_state_hash(_moe_layer(ep_size=2, use_ep=True))
    assert tp and ep and tp != ep, (tp, ep)
    assert _extract_module_state_hash(_moe_layer(tp_size=2)) == tp  # stable


def test_derive_op_dtype():
    assert _derive_op_dtype([_inp("long int", (65,)), _inp("c10::BFloat16")]) == "bfloat16"  # rotary: positions first
    assert _derive_op_dtype([_inp("c10::Half")]) == "float16"
    assert _derive_op_dtype([_inp("torch.float32")]) == "float32"
    assert _derive_op_dtype([_inp("float")]) == "float32"
    assert _derive_op_dtype([_inp("int", (7,)), _inp("long int", (7,))]) == "int32"
    assert _derive_op_dtype([_inp("c10::Float8_e4m3fn")]) == "float8_e4m3fn"
    assert _derive_op_dtype([_inp("Scalar", (), "Scalar")], fallback="bfloat16") == "bfloat16"
    assert _derive_op_dtype([], fallback="") == ""


# A custom operator registered in this process: it resolves through torch.ops
# like a loaded vLLM op, but PyTorch does not define it.
_TEST_LIB = torch.library.Library("doolysigtest", "DEF")
_TEST_LIB.define("scale(Tensor x, str layer_name) -> Tensor")


def _tensor(dtype, *dims):
    """A traced tensor input: the first dim is the token count, the rest MODEL_CONFIG."""
    return ProfileInput(type="Tensor", dtype=dtype,
                        dimensions=[DimensionType.NUM_TOKENS] + [DimensionType.MODEL_CONFIG] * len(dims),
                        actual_shape=[470, *dims])


def _scalar(value, dtype="Scalar"):
    return ProfileInput(type="Scalar", dtype=dtype, dimensions=[], actual_shape=[], scalar_value=value)


def _raw_op(op_name, parent_state, inputs):
    """A raw-op anchor (not a module) under a module whose state hash is ``parent_state``."""
    from doolyprof.profiler.module import ModuleInfo
    from doolyprof.profiler.parser import Event, OperationHierarchy
    from doolyprof.profiler.trace import _derive_op_dtype

    def ev(name, cat, **args):
        return Event(cat=cat, name=name, ts=0, dur=1, ev_idx=None, external_id=None, python_id=None,
                     python_parent_id=None, args=args)

    parent = ev("KimiDecoderLayer", "module", **({"state_hash": parent_state} if parent_state else {}))
    h = OperationHierarchy(anchor=ev(op_name, "cpu_op"), root_module=parent, nearest_module=parent,
                           kernels=[], taints=[])
    return ModuleInfo(module_name=op_name, operation_name=op_name, inputs=inputs,
                      dtype=_derive_op_dtype(inputs), hierarchy=h)


def test_is_builtin_operator():
    from doolyprof.profiler.module import is_builtin_operator
    for name in ("aten::pad", "aten::mul.Tensor", "aten::mul", "aten::add.Tensor", "aten::linear"):
        assert is_builtin_operator(name), name
    for name in ("doolysigtest::scale",             # registered, but not by PyTorch
                 "vllm::unified_attention_with_output", "_C::rms_norm",  # custom ops
                 "aten::no_such_op", "KimiDecoderLayer", "", "::pad"):
        assert not is_builtin_operator(name), name


def test_builtin_op_ignores_parent_state():
    """A builtin PyTorch op called identically under differently configured modules is one op
    (Kimi-K3's sequence-parallel pad under 16 decoder-layer states); its own arguments still
    separate calls that really differ."""
    from doolyprof.profiler.module import ModuleComparator
    cmp = ModuleComparator()
    sig = lambda m: cmp._hash_signature(cmp._module_signature(m))  # noqa: E731
    pad = lambda state, p="[0, 0, 0, 2]": _raw_op(  # noqa: E731
        "aten::pad", state, [_tensor("c10::BFloat16", 7168), _scalar(p, "ScalarList")])
    assert sig(pad("0a1b2c3d4e5f")) == sig(pad("ffffffffffff")) == sig(pad(""))
    assert sig(pad("0a1b2c3d4e5f")) != sig(pad("0a1b2c3d4e5f", "[0, 0, 0, 6]"))


def test_custom_op_keeps_parent_state():
    """A custom op can read its layer's configuration through the layer name (vLLM attention's
    sliding window), so the same call under differently configured layers stays apart."""
    from doolyprof.profiler.module import ModuleComparator
    cmp = ModuleComparator()
    sig = lambda m: cmp._hash_signature(cmp._module_signature(m))  # noqa: E731
    for op in ("doolysigtest::scale", "vllm::unified_attention_with_output"):
        call = lambda state: _raw_op(op, state, [_tensor("c10::BFloat16", 2048), _scalar("layers.0.attn")])  # noqa: E731
        assert sig(call("0a1b2c3d4e5f")) != sig(call("ffffffffffff")), op
        assert sig(call("0a1b2c3d4e5f")) == sig(call("0a1b2c3d4e5f")), op


def test_every_input_dtype_in_signature():
    """The dtype field is the first floating input's; a different dtype in any other input is a
    different op."""
    from doolyprof.profiler.module import ModuleComparator
    cmp = ModuleComparator()
    sig = lambda m: cmp._hash_signature(cmp._module_signature(m))  # noqa: E731
    bf16_scale = _raw_op("aten::mul.Tensor", "", [_tensor("c10::BFloat16", 2816), _tensor("c10::BFloat16")])
    fp32_scale = _raw_op("aten::mul.Tensor", "", [_tensor("c10::BFloat16", 2816), _tensor("float")])
    assert bf16_scale.dtype == fp32_scale.dtype == "bfloat16"
    assert bf16_scale.input_dtypes() == ("bfloat16", "bfloat16")
    assert fp32_scale.input_dtypes() == ("bfloat16", "float32")
    assert sig(bf16_scale) != sig(fp32_scale)
    # index dtypes count too
    i32 = _raw_op("aten::index_select", "", [_tensor("c10::BFloat16", 2048), _scalar("0"), _tensor("int")])
    i64 = _raw_op("aten::index_select", "", [_tensor("c10::BFloat16", 2048), _scalar("0"), _tensor("long int")])
    assert i32.input_dtypes() == ("bfloat16", "", "int32") and sig(i32) != sig(i64)
    # spellings of one dtype agree, and an omitted trailing argument is no difference
    a = _raw_op("aten::add.Tensor", "", [_tensor("c10::BFloat16", 2048), _tensor("torch.bfloat16", 2048)])
    b = _raw_op("aten::add.Tensor", "", [_tensor("c10::BFloat16", 2048), _tensor("c10::BFloat16", 2048), _scalar(None)])
    assert a.input_dtypes() == b.input_dtypes() == ("bfloat16", "bfloat16")
    assert sig(a) == sig(b)


def test_dedup_groups_follow_signature():
    """The parser's dedup groups apply the same rule as the signature: the builtin op's calls
    under two layer states become one entry counting both, the custom op's stay two."""
    from doolyprof.profiler.trace import TraceParser
    infos = [_raw_op("aten::pad", s, [_tensor("c10::BFloat16", 7168), _scalar("[0, 0, 0, 2]", "ScalarList")])
             for s in ("0a1b2c3d4e5f", "ffffffffffff")]
    infos += [_raw_op("doolysigtest::scale", s, [_tensor("c10::BFloat16", 2048), _scalar("layers.0.attn")])
              for s in ("0a1b2c3d4e5f", "ffffffffffff")]
    out = TraceParser._deduplicate_module_infos(infos)
    counts = sorted((d.operation_name, d.count) for d in out)
    assert counts == [("aten::pad", 2), ("doolysigtest::scale", 1), ("doolysigtest::scale", 1)], counts


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
