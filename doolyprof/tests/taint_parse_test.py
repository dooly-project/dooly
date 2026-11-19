"""Unit tests for taint-string dimension parsing and trace batch selection.

* ``parse_taint_string`` / ``extract_inputs_from_taint`` must keep every traced
  dimension, including ``SINGLETON(1)`` broadcast dims, so dimension types stay
  aligned with the traced shape (a dropped dim shifted MODEL_CONFIG padding onto
  the token axis and leaked the trace batch size into signatures).
* A TensorList argument (``aten::cat``'s tensors) is one trace input holding a
  list of shapes but one taint segment per tensor; the arguments after it must
  stay aligned and the signature parameters must stay hashable.
* ``choose_trace_batch_size`` must avoid every registry collision the scheduler
  would otherwise raise as a TAINT CONFLICT.

Runs under pytest or directly: ``python doolyprof/tests/taint_parse_test.py``. CPU only.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from doolyprof.profiler.taint import DimensionType, parse_taint_string  # noqa: E402
from doolyprof.profiler.finput import extract_inputs_from_taint  # noqa: E402
from doolyprof.profiler.parser import Event  # noqa: E402
from doolyprof.profiler.module import ModuleInfo  # noqa: E402
from doolyprof.tracer.registry import choose_trace_batch_size, choose_trace_batch  # noqa: E402

MC, NT, NR, MIX = (DimensionType.MODEL_CONFIG, DimensionType.NUM_TOKENS,
                   DimensionType.NUM_REQUESTS, DimensionType.MIX)


def _types(taint_input):
    return [d.dim_type for d in taint_input.dimensions]


def test_singleton_dim_is_kept():
    s = "aten::masked_fill_.Scalar IN:[[NUM_TOKS(504), MODEL_CONFIG(2304)] | [NUM_TOKS(504), SINGLETON(1)] | []]"
    inputs = parse_taint_string(s)
    assert len(inputs) == 3
    assert _types(inputs[0]) == [NT, MC]
    assert _types(inputs[1]) == [NT, MC], _types(inputs[1])  # SINGLETON kept, as a fixed dim
    assert [d.value for d in inputs[1].dimensions] == [504, 1]
    assert inputs[2].dimensions == []


def test_singleton_keeps_token_axis_aligned_with_trace_shape():
    s = "aten::masked_fill_.Scalar IN:[[NUM_TOKS(504), MODEL_CONFIG(2304)] | [NUM_TOKS(504), SINGLETON(1)] | []]"
    profile_inputs = extract_inputs_from_taint(
        s, trace_input_dims=[[504, 2304], [504, 1], []],
        trace_input_dtypes=["c10::BFloat16", "bool", "Scalar"], concrete_inputs=["", "", "0.0"])
    mask = profile_inputs[1]
    assert mask.type == "Tensor" and mask.actual_shape == [504, 1]
    assert mask.dimensions == [NT, MC], mask.dimensions  # before: [MC, NT] -> 504 became a "model constant"


def test_mla_kv_tensor_with_singleton_middle_dim():
    s = "vllm::unified_mla_kv_cache_update IN:[[NUM_TOKS(36)[hist:NUM_TOKS:36], MODEL_CONFIG(512)] | [NUM_TOKS(36)[hist:NUM_TOKS:36], SINGLETON(1), MODEL_CONFIG(64)] | []]"
    inputs = parse_taint_string(s)
    assert _types(inputs[0]) == [NT, MC]
    assert _types(inputs[1]) == [NT, MC, MC]
    assert [d.value for d in inputs[1].dimensions] == [36, 1, 64]


def test_known_labels_and_max_prefix():
    inputs = parse_taint_string("op IN:[[MAX_NUM_TOKS(521), MAX_NUM_REQS(263), NUM_REQS(7), ?(9), MODEL_CONFIG(128)]]")
    assert _types(inputs[0]) == [NT, NR, NR, MC, MC]


def test_mix_history_tracer_order_and_legacy_order():
    # tracer writes taint:qty (tensor._format_dim); older strings had qty:taint
    d = parse_taint_string("op IN:[[MIX(8064)[hist:NUM_TOKS:504,MODEL_CONFIG:16], MODEL_CONFIG(128)]]")[0].dimensions[0]
    assert d.dim_type == MIX and d.value == 8064 and d.components == {"NUM_TOKS": 504, "MODEL_CONFIG": 16}, d
    d = parse_taint_string("op IN:[[MIX(240)[hist:6:NUM_TOKS,40:MODEL_CONFIG], MODEL_CONFIG(128)]]")[0].dimensions[0]
    assert d.dim_type == MIX and d.value == 240 and d.components == {"NUM_TOKS": 6, "MODEL_CONFIG": 40}
    d = parse_taint_string("op IN:[[MIX(1077248)[hist:MAX_NUM_REQS:1,MODEL_CONFIG:1]]]")[0].dimensions[0]
    assert d.components == {"MAX_NUM_REQS": 1, "MODEL_CONFIG": 1}


def test_mix_component_axis_is_exact_label():
    from doolyprof.profiler.input_generator import _component_axis
    assert _component_axis("NUM_TOKS") == "num_tokens"
    assert _component_axis("NUM_REQS") == "num_requests"
    assert _component_axis("MAX_NUM_REQS") is None     # buffer bound: fixed, not swept
    assert _component_axis("MAX_NUM_TOKS") is None
    assert _component_axis("MODEL_CONFIG") is None and _component_axis("?") is None


def _mix_module(taint, dims):
    from doolyprof.profiler.module import ModuleInfo
    inputs = extract_inputs_from_taint(taint, trace_input_dims=dims, trace_input_dtypes=["c10::BFloat16"] * len(dims))
    return ModuleInfo(module_name="synthetic", operation_name="aten::mul.Tensor", inputs=inputs, dtype="bfloat16")


def test_generator_sweeps_mix_dimension_with_tokens():
    """End-to-end through the real generator on GPU: a flattened tokens x heads tensor
    (MIX = NUM_TOKS * 16) must follow the token sweep, and the op must run on the result."""
    import torch
    if not torch.cuda.is_available():
        print("SKIP (no CUDA)"); return
    from doolyprof.profiler.input_generator import InputGenerator
    taint = ("aten::mul.Tensor IN:[[MIX(8064)[hist:NUM_TOKS:504,MODEL_CONFIG:16], MODEL_CONFIG(128)] | "
             "[MIX(8064)[hist:NUM_TOKS:504,MODEL_CONFIG:16], MODEL_CONFIG(128)]]")
    module = _mix_module(taint, [[8064, 128], [8064, 128]])
    assert module.inputs[0].dimensions == [MIX, MC] and module.inputs[0].dimension_components[0] == {"NUM_TOKS": 504, "MODEL_CONFIG": 16}
    gen = InputGenerator(module, max_num_request=4, max_num_token=256, test_counts=3)
    configs = gen.prepare_inputs(mode="vidur", use_batch_config=False)
    assert configs and len({c["num_tokens"] for c in configs}) > 1, configs  # the token axis is swept
    for cfg in configs:
        tensors, params = gen._create_tensor_inputs_from_params(cfg)
        expect = (params["num_tokens"] * 16, 128)
        assert tuple(tensors[0].shape) == expect and tuple(tensors[1].shape) == expect, (cfg, tensors[0].shape)
        out = torch.ops.aten.mul.Tensor(tensors[0], tensors[1])
        assert tuple(out.shape) == expect
    torch.cuda.synchronize()


def test_generator_keeps_max_bound_mix_fixed():
    import torch
    if not torch.cuda.is_available():
        print("SKIP (no CUDA)"); return
    from doolyprof.profiler.input_generator import InputGenerator
    taint = "aten::mul.Tensor IN:[[MIX(538624)[hist:MAX_NUM_REQS:263,MODEL_CONFIG:2048]] | [MIX(538624)[hist:MAX_NUM_REQS:263,MODEL_CONFIG:2048]]]"
    module = _mix_module(taint, [[538624], [538624]])
    gen = InputGenerator(module, max_num_request=4, max_num_token=256, test_counts=3)
    for cfg in ({"num_tokens": 8, "num_requests": 1}, {"num_tokens": 256, "num_requests": 4}):
        tensors, _ = gen._create_tensor_inputs_from_params(cfg)
        assert tuple(tensors[0].shape) == (538624,), tensors[0].shape  # MAX_* bound: never swept


def test_choose_trace_batch_respects_kv_capacity():
    # vLLM 0.33 Kimi-K3 (fake TP8 x DP8 on one A100): 3.2 GiB of KV cache in
    # 39 large hybrid blocks; the 94-request batch ran 39 requests in its first
    # step and NUM_REQS=39 hit a MODEL_CONFIG value mid-generate.
    from doolyprof.tracer.registry import kv_cache_capacity
    reg = {v: "MODEL_CONFIG" for v in range(2, 94)}
    cap = kv_cache_capacity(39, 1856, 1)
    assert cap(5) == 39
    try:
        choose_trace_batch(5, 12, 263, 521, reg, kv_capacity=cap)
    except ValueError as e:
        assert "hold at most 39 requests" in str(e), e
        # Kimi-K3's layer-index lists cover 1..93: 94 requests is the first clear count
        assert "smallest request count clear of the registry is 94" in str(e), e
    else:
        raise AssertionError("expected no admissible batch")
    assert choose_trace_batch(5, 12, 263, 521, {}, kv_capacity=cap) == (5, 12)
    assert kv_cache_capacity(None, 16) is None


def _reshape_module(taint, concrete):
    inputs = extract_inputs_from_taint(taint, trace_input_dims=[[60, 12, 256], []],
                                       trace_input_dtypes=["c10::BFloat16", "ScalarList"],
                                       concrete_inputs=["", concrete])
    return ModuleInfo(module_name="synthetic", operation_name="aten::reshape", inputs=inputs, dtype="bfloat16")


def test_shape_list_follows_token_sweep():
    # vLLM 0.17.1 Qwen3.5 attention: q.reshape(*q.shape[:-1], -1) reaches the op
    # as [60, -1] (the traced token count as a plain int). Kept fixed, every
    # token count that does not divide 60 * 3072 failed with "only one
    # dimension can be inferred".
    from doolyprof.profiler.input_generator import InputGenerator
    taint = "aten::reshape IN:[[NUM_TOKS(60), MODEL_CONFIG(12), MODEL_CONFIG(256)]]"
    gen = InputGenerator(_reshape_module(taint, "[60, -1]"))
    assert gen._sweep_int_list([60, -1], 4, 1) == [4, -1]
    assert gen._sweep_int_list((60, 12, 256), 7, 1) == (7, 12, 256)
    assert gen._sweep_int_list([-1, 256], 4, 1) == [-1, 256]   # fixed dims stay


def test_shape_list_ambiguous_value_stays():
    # 12 is the traced token count and also a fixed dim of the same input: leave it
    from doolyprof.profiler.input_generator import InputGenerator
    taint = "aten::reshape IN:[[NUM_TOKS(12), MODEL_CONFIG(12), MODEL_CONFIG(256)]]"
    inputs = extract_inputs_from_taint(taint, trace_input_dims=[[12, 12, 256], []],
                                       trace_input_dtypes=["c10::BFloat16", "ScalarList"],
                                       concrete_inputs=["", "[12, -1]"])
    gen = InputGenerator(ModuleInfo(module_name="synthetic", operation_name="aten::reshape", inputs=inputs, dtype="bfloat16"))
    assert gen._sweep_int_list([12, -1], 4, 1) == [12, -1]


def test_event_dims_from_taint_string_keeps_singleton():
    ev = Event(cat="cpu_op", name="x", ts=0, dur=1, ev_idx=None, external_id=None, python_id=None, python_parent_id=None)
    dims = ev.get_dims_from_taint_string("[[NUM_TOKS(504), MODEL_CONFIG(2304)] | [NUM_TOKS(504), SINGLETON(1)]]")
    assert dims == [[504, 2304], [504, 1]], dims


def test_choose_trace_batch_size_kimi_linear():
    # Kimi-Linear registers 2..27 (kda_layers + full_attn_layers) as MODEL_CONFIG;
    # 18-token prompt, max_num_seqs=263, max_num_batched_tokens=521.
    reg = {v: "MODEL_CONFIG" for v in range(2, 28)}
    reg.update({32: "MODEL_CONFIG", 263: "MAX_NUM_REQS", 521: "MAX_NUM_TOKS", 512: "MODEL_CONFIG"})
    assert choose_trace_batch_size(7, 18, 263, 521, reg) == 28   # 2..27 taken; 28*18=504 fits one step
    assert choose_trace_batch_size(2, 18, 263, 521, reg) == 28
    assert choose_trace_batch_size(28, 18, 263, 521, reg) == 28
    assert choose_trace_batch_size(40, 18, 263, 521, reg) == 28  # 29*18 > 521: fall back downward


def test_choose_trace_batch_truncates_prompt_when_needed():
    reg = {v: "MODEL_CONFIG" for v in range(2, 28)}
    reg.update({32: "MODEL_CONFIG", 263: "MAX_NUM_REQS", 521: "MAX_NUM_TOKS", 512: "MODEL_CONFIG"})
    # default 47-token prompt: 521 // 47 = 11 and 2..27 are taken -> shorten to 18 tokens, batch 28
    assert choose_trace_batch(7, 47, 263, 521, reg) == (28, 18)
    assert choose_trace_batch(7, 18, 263, 521, reg) == (28, 18)   # nothing to truncate
    assert choose_trace_batch(7, 19, 263, 521, {}) == (7, 19)     # empty registry: as requested
    big = {v: "MODEL_CONFIG" for v in range(2, 300)}
    try:
        choose_trace_batch(7, 47, 263, 521, big)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_choose_trace_batch_size_derived_values():
    assert choose_trace_batch_size(5, 10, 263, 521, {6: "MODEL_CONFIG"}) == 7    # 5 -> cu_seqlens 6 collides; 6 collides itself
    assert choose_trace_batch_size(5, 10, 263, 521, {50: "MODEL_CONFIG"}) == 6   # 5*10 tokens collides
    assert choose_trace_batch_size(5, 10, 263, 521, {}) == 5                     # nothing to avoid
    try:
        choose_trace_batch_size(2, 300, 263, 521, {})                             # no batch fits one step
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_module_annotation_path_and_state_parsed():
    from doolyprof.profiler.parser import HierarchicalTraceParser
    name = ("MODULE: GemmaRMSNorm IN:[[NUM_TOKS(504), MODEL_CONFIG(16), MODEL_CONFIG(256)]] "
            "DTYPES:[torch.bfloat16] STATE:4fc82af4562a PATH:language_model.model.layers.3.self_attn.q_norm")
    pr = HierarchicalTraceParser(events=[{"ph": "X", "cat": "user_annotation", "name": name, "ts": 0, "dur": 5,
                                          "pid": 1, "tid": 1, "args": {}}])
    pr._parse_events(pr.raw_events)
    ev = pr.modules[0]
    assert ev.name == "GemmaRMSNorm", ev.name
    assert ev.args["state_hash"] == "4fc82af4562a"
    assert ev.args["module_path"] == "language_model.model.layers.3.self_attn.q_norm"
    assert "PATH" not in (ev.params or "") and "STATE" not in (ev.params or "")


def _resolver_on(model):
    from doolyprof.profiler.resolver import ModuleResolver
    r = ModuleResolver.__new__(ModuleResolver)
    r.vlp = type("V", (), {"model": model})()
    return r


def test_recorded_path_group_check():
    import torch
    from doolyprof.profiler.module import ModuleInfo
    from doolyprof.tracer.hooks import _extract_module_state_hash

    class Norm(torch.nn.Module):
        def __init__(self, n, eps=1e-6):
            super().__init__(); self.weight = torch.nn.Parameter(torch.ones(n)); self.eps = eps

    class Attn(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.q_norm = Norm(256); self.k_norm = Norm(256)

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.input_layernorm = Norm(2560); self.self_attn = Attn()

    model = torch.nn.Module(); model.layers = torch.nn.ModuleList([Layer(), Layer()])
    r = _resolver_on(model)
    st = _extract_module_state_hash(model.layers[0].self_attn.q_norm)
    ok = ModuleInfo(module_name="Norm", operation_name="Norm", inputs=[], is_module=True, state_hash=st,
                    instance_paths=["layers.1.self_attn.q_norm", "layers.0.self_attn.q_norm", "layers.0.self_attn.k_norm"])
    assert r._pick_recorded_path("Norm", ok) == "layers.1.self_attn.q_norm"
    # a group that wrongly mixes 256- and 2560-wide norms is detected (same state hash, different weight)
    mixed = ModuleInfo(module_name="Norm", operation_name="Norm", inputs=[], is_module=True, state_hash=st,
                       instance_paths=["layers.0.input_layernorm", "layers.0.self_attn.q_norm"])
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r._pick_recorded_path("Norm", mixed)
    assert "mixes 2 different instance kinds" in buf.getvalue(), buf.getvalue()
    # a path that does not exist is reported, the remaining member is used
    missing = ModuleInfo(module_name="Norm", operation_name="Norm", inputs=[], is_module=True, state_hash=st,
                         instance_paths=["layers.9.self_attn.q_norm", "layers.0.self_attn.q_norm"])
    with contextlib.redirect_stdout(io.StringIO()):
        assert r._pick_recorded_path("Norm", missing) == "layers.0.self_attn.q_norm"


_CAT_TAINT = "aten::cat IN:[[SINGLETON(1)] | [NUM_REQS(28)]]"
_CAT_DIMS = [[[1], [28]], []]


def test_tensor_list_input_aligned():
    inputs = extract_inputs_from_taint(_CAT_TAINT, trace_input_dims=_CAT_DIMS,
                                       trace_input_dtypes=["TensorList", "Scalar"],
                                       concrete_inputs=["", "0"])
    assert [i.type for i in inputs] == ["TensorList", "Scalar"], [i.type for i in inputs]
    elems = inputs[0].tensor_parts()
    assert [e.actual_shape for e in elems] == [[1], [28]], [e.actual_shape for e in elems]
    assert elems[1].dimensions == [DimensionType.NUM_REQUESTS], elems[1].dimensions
    assert inputs[1].scalar_value == "0", inputs[1]
    info = ModuleInfo(module_name="KDA", operation_name="aten::cat", inputs=inputs, dtype="int")
    params = info._semantic_params_list()[0]
    hash(tuple(sorted(params.items())))  # dedup key must be hashable
    assert params == {"input0_t0_dim0": 1, "input1_scalar": 0.0}, params
    rt = type(inputs[0]).from_dict(inputs[0].to_dict())
    assert [e.actual_shape for e in rt.tensor_parts()] == [[1], [28]]


def test_generator_builds_tensor_list_on_request_sweep():
    import torch
    if not torch.cuda.is_available():
        print("SKIP (no CUDA)"); return
    from doolyprof.profiler.input_generator import InputGenerator
    inputs = extract_inputs_from_taint(_CAT_TAINT, trace_input_dims=_CAT_DIMS,
                                       trace_input_dtypes=["TensorList", "Scalar"],
                                       concrete_inputs=["", "0"])
    module = ModuleInfo(module_name="KDA", operation_name="aten::cat", inputs=inputs, dtype="bfloat16")
    gen = InputGenerator(module, max_num_request=4, max_num_token=256, test_counts=3)
    configs = gen.prepare_inputs(mode="vidur", use_batch_config=False)
    assert configs and len({c["num_requests"] for c in configs}) > 1, configs
    for cfg in configs:
        tensors, params = gen._create_tensor_inputs_from_params(cfg)
        assert [tuple(t.shape) for t in tensors[0]] == [(1,), (params["num_requests"],)], tensors
        out = torch.cat(tensors[0], tensors[1])
        assert tuple(out.shape) == (1 + params["num_requests"],)


def test_offset_label_parsed_and_swept():
    import torch
    from doolyprof.profiler.input_generator import InputGenerator
    t = parse_taint_string("op IN:[[NUM_REQS+1(6)] | [NUM_TOKS-1(59), MODEL_CONFIG(8)] | [MAX_NUM_REQS+1(264)]]")
    assert [(d.dim_type, d.value, d.offset) for d in t[0].dimensions] == [(NR, 6, 1)]
    assert [(d.dim_type, d.value, d.offset) for d in t[1].dimensions] == [(NT, 59, -1), (MC, 8, 0)]
    assert [(d.dim_type, d.offset) for d in t[2].dimensions] == [(NR, 1)]
    taint = "op IN:[[NUM_REQS+1(6)] | [NUM_TOKS(60), MODEL_CONFIG(8)]]"
    inputs = extract_inputs_from_taint(taint, trace_input_dims=[[6], [60, 8]],
                                       trace_input_dtypes=["c10::BFloat16", "c10::BFloat16"],
                                       concrete_inputs=["", ""])
    module = ModuleInfo(module_name="M", operation_name="aten::foo", inputs=inputs, dtype="bfloat16")
    # the offset is part of what the op computes on, so it is in the signature
    assert module._semantic_params_list()[0].get("input0_dim0_offset") == 1
    gen = InputGenerator(module, max_num_request=4, max_num_token=256, test_counts=3)
    tensors, _ = gen._create_tensor_inputs_from_params({"num_tokens": 32, "num_requests": 3})
    assert [tuple(x.shape) for x in tensors] == [(4,), (32, 8)], [tuple(x.shape) for x in tensors]
    assert gen._sweep_int_list([6, 60, 8], 32, 3) == [4, 32, 8]


def test_gathered_comm_dim_is_tokens_times_world_size():
    # Qwen1.5-MoE on DP2: the combine's input holds both ranks' 72 tokens
    from doolyprof.profiler.utils.comm_emulation import label_gathered_dims
    shapes = ["[?(144), MODEL_CONFIG(2048)]", "[NUM_TOKS(72), MODEL_CONFIG(2048)]"]
    out = label_gathered_dims(shapes, 2)
    assert out == ["[MIX(144)[hist:NUM_TOKS:72,WORLD_SIZE:2], MODEL_CONFIG(2048)]", shapes[1]], out
    t = parse_taint_string(f"COMM: combine IN:[{' | '.join(out)}]")
    d = t[0].dimensions[0]
    assert d.dim_type == DimensionType.MIX and d.components == {"NUM_TOKS": 72, "WORLD_SIZE": 2}, d
    assert label_gathered_dims(shapes, 1) == shapes                     # no group, nothing gathered
    assert label_gathered_dims(shapes + ["[NUM_REQS(72)]"], 2) == shapes + ["[NUM_REQS(72)]"]  # ambiguous
    assert label_gathered_dims(["[?(100)]", "[NUM_TOKS(72)]"], 2) == ["[?(100)]", "[NUM_TOKS(72)]"]
    # the profiler sizes such a dim as tokens x world size
    from doolyprof.profiler.input_generator import InputGenerator
    inputs = extract_inputs_from_taint(taint_string=f"COMM: combine IN:[{out[0]}]",
                                       trace_input_dims=[[144, 2048]], trace_input_dtypes=["c10::BFloat16"])
    module = ModuleInfo(module_name="combine", operation_name="combine", inputs=inputs)
    gen = InputGenerator(module, max_num_request=4, max_num_token=256, test_counts=3)
    tensors, _ = gen._create_tensor_inputs_from_params({"num_tokens": 32, "num_requests": 3})
    assert tuple(tensors[0].shape) == (64, 2048), tuple(tensors[0].shape)


def _paged_batch(block_size=16, num_blocks=50, reqs=3, kv=37, new=5, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    per_seq = -(-(kv + new) // block_size)
    rows = [rng.integers(0, num_blocks, size=per_seq).tolist() for _ in range(reqs)]
    req_indices = np.repeat(np.arange(reqs, dtype=np.int32), new)
    positions = np.tile(np.arange(kv, kv + new, dtype=np.int32), reqs)
    return rows, req_indices, positions


def test_paged_block_table_slots_follow_the_cache_layout():
    import torch
    from doolyprof.profiler.utils.vllm_compat import make_block_table, fill_slot_mapping
    rows, req_indices, positions = _paged_batch()
    bt = make_block_table(block_size=16, max_num_reqs=8, max_num_blocks_per_req=10,
                          max_num_batched_tokens=64, device="cpu", pin_memory=False)
    bt.clear()
    for r, ids in enumerate(rows):
        bt.add_row(ids, row_idx=r)
    fill_slot_mapping(bt, len(rows), req_indices, positions)
    expect = [rows[r][p // 16] * 16 + p % 16 for r, p in zip(req_indices, positions)]
    assert bt.slot_mapping.gpu[:len(expect)].tolist() == expect
    assert bt.block_table.gpu[:3, :len(rows[0])].tolist() == rows
    assert bt.block_table.gpu.dtype == torch.int32 and bt.slot_mapping.gpu.dtype == torch.int64


def test_paged_block_table_matches_vllm_block_table():
    """Where the installed vLLM still has its runner's BlockTable (0.17-0.33)
    and a GPU is available, the slots agree with the ones it computes."""
    import inspect
    import torch
    try:
        from vllm.v1.worker.block_table import BlockTable
    except ImportError:
        print("SKIP (this vLLM has no vllm.v1.worker.block_table)"); return
    if not torch.cuda.is_available():
        print("SKIP (no GPU)"); return
    from doolyprof.profiler.utils.vllm_compat import call_with_supported_kwargs, make_block_table, fill_slot_mapping
    rows, req_indices, positions = _paged_batch()
    kw = dict(block_size=16, max_num_reqs=8, max_num_blocks_per_req=10, max_num_batched_tokens=64,
              pin_memory=True, device=torch.device("cuda"), kernel_block_size=16, cp_kv_cache_interleave_size=1)
    theirs = call_with_supported_kwargs(BlockTable, optional=("kernel_block_size", "cp_kv_cache_interleave_size"), **kw)
    ours = make_block_table(**kw)
    for bt in (theirs, ours):
        bt.clear()
        for r, ids in enumerate(rows):
            bt.add_row(ids, row_idx=r)
    n = len(positions)
    if "query_start_loc" in inspect.signature(theirs.compute_slot_mapping).parameters:
        qsl = torch.arange(0, n + 1, n // len(rows), dtype=torch.int32, device="cuda")
        theirs.commit_block_table(len(rows))
        theirs.compute_slot_mapping(len(rows), qsl, torch.as_tensor(positions, dtype=torch.int64, device="cuda"))
    else:
        theirs.compute_slot_mapping(req_indices, positions)
        theirs.commit_block_table(len(rows))
        theirs.commit_slot_mapping(n)
    fill_slot_mapping(ours, len(rows), req_indices, positions)
    assert ours.slot_mapping.gpu[:n].tolist() == theirs.slot_mapping.gpu[:n].tolist()
    assert ours.block_table.gpu[:len(rows)].tolist() == theirs.block_table.gpu[:len(rows)].tolist()


def test_module_called_with_none_for_unrecorded_required_arguments():
    """A decoder layer's first call passes residual=None, which the trace does
    not record; vLLM 0.34 makes residual a required parameter."""
    import torch
    from doolyprof.profiler.utils.vllm_compat import call_module_with_recorded_arguments

    class Layer(torch.nn.Module):
        def forward(self, positions, hidden_states, residual, *, scale=1.0):
            return positions, hidden_states, residual, scale

    class OldLayer(torch.nn.Module):
        def forward(self, positions, hidden_states, residual=0):
            return residual

    x, pos = torch.zeros(3, 4), torch.arange(3)
    out = call_module_with_recorded_arguments(Layer(), [], {"positions": pos, "hidden_states": x})
    assert out[2] is None and out[3] == 1.0 and out[1] is x
    out = call_module_with_recorded_arguments(Layer(), [pos], {"hidden_states": x})
    assert out[0] is pos and out[2] is None
    res = torch.ones(3, 4)
    assert call_module_with_recorded_arguments(Layer(), [pos, x, res], {})[2] is res
    assert call_module_with_recorded_arguments(OldLayer(), [], {"positions": pos, "hidden_states": x}) == 0


def test_topology_label_from_gpu_model_and_links():
    # collective measurements were stored and looked up under a hard-coded
    # host name ("mew1") on every machine
    from doolyprof.profiler.utils.topology import topology_label, NO_GPU
    assert topology_label("NVIDIA A100-SXM4-80GB", True) == "nvidia-a100-sxm4-80gb-nvlink"
    assert topology_label("NVIDIA A100 80GB PCIe", False) == "nvidia-a100-80gb-pcie"     # no "-pcie-pcie"
    assert topology_label("NVIDIA A100 80GB PCIe", True) == "nvidia-a100-80gb-pcie-nvlink"
    assert (topology_label("NVIDIA RTX PRO 6000 Blackwell Server Edition", False)
            == "nvidia-rtx-pro-6000-blackwell-server-edition-pcie")
    assert topology_label("NVIDIA H100 80GB HBM3", None) == "nvidia-h100-80gb-hbm3-unknown-link"
    assert topology_label(None, None) == NO_GPU and topology_label("", True) == NO_GPU


def test_machine_topology_prefers_the_environment():
    from doolyprof.profiler.utils import topology
    saved = os.environ.get("DOOLY_TOPOLOGY")
    try:
        os.environ["DOOLY_TOPOLOGY"] = "mew1"          # measurements stored under an older label
        assert topology.machine_topology() == "mew1"
        os.environ.pop("DOOLY_TOPOLOGY")
        derived = topology.machine_topology()
        assert derived and derived == topology._derived_topology() and "mew" not in derived, derived
    finally:
        if saved is None:
            os.environ.pop("DOOLY_TOPOLOGY", None)
        else:
            os.environ["DOOLY_TOPOLOGY"] = saved


# nvidia-smi topo -m on a machine of NVLink pairs (A100 PCIe with bridges) and on a PCIe-only one
_TOPO_NV_PAIRS = (
    "\t\x1b[4mGPU0\tGPU1\tGPU2\tGPU3\tNIC0\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\x1b[0m\n"
    "GPU0\t X \tNV12\tSYS\tSYS\tNODE\t0,2,4\t0\t\tN/A\n"
    "GPU1\tNV12\t X \tSYS\tSYS\tNODE\t0,2,4\t0\t\tN/A\n"
    "GPU2\tSYS\tSYS\t X \tNV12\tSYS\t1,3,5\t1\t\tN/A\n"
    "GPU3\tSYS\tSYS\tNV12\t X \tSYS\t1,3,5\t1\t\tN/A\n"
    "NIC0\tNODE\tNODE\tSYS\tSYS\t X \t\t\t\t\n\nLegend:\n\n  X    = Self\n"
    "  NV#  = Connection traversing a bonded set of # NVLinks\n")
_TOPO_PCIE = (
    "\t\x1b[4mGPU0\tGPU1\tGPU2\tGPU3\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\x1b[0m\n"
    "GPU0\t X \tPIX\tNODE\tNODE\t0-95\t0\t\tN/A\n"
    "GPU1\tPIX\t X \tNODE\tNODE\t0-95\t0\t\tN/A\n"
    "GPU2\tNODE\tNODE\t X \tPIX\t0-95\t0\t\tN/A\n"
    "GPU3\tNODE\tNODE\tPIX\t X \t0-95\t0\t\tN/A\n")


def test_gpu_pick_takes_the_best_linked_free_gpus():
    # the docker wrapper took the first free GPUs, so a 2-rank collective could be
    # measured on GPUs of two different NVLink pairs and stored as NVLink latencies
    from doolyprof.profiler.utils.topology import parse_topo_matrix, pick_gpus, same_class
    links = parse_topo_matrix(_TOPO_NV_PAIRS)
    assert set(links) == {0, 1, 2, 3} and links[0][1] == "NV12" and links[1][2] == "SYS", links
    assert pick_gpus(2, [0, 1, 2, 3], links) == ([0, 1], "NV12", "NV12")
    assert pick_gpus(2, [1, 2, 3], links) == ([2, 3], "NV12", "NV12")   # not 1,2: the free pair
    chosen, link, best = pick_gpus(2, [1, 2], links)
    assert (chosen, link, best) == ([1, 2], "SYS", "NV12") and not same_class(link, best)
    assert pick_gpus(4, [0, 1, 2, 3], links) == ([0, 1, 2, 3], "SYS", "SYS")
    assert pick_gpus(4, [0, 1, 2], links) is None
    assert pick_gpus(1, [3], links) == ([3], "-", "-")
    pcie = parse_topo_matrix(_TOPO_PCIE)
    assert pick_gpus(2, [1, 2, 3], pcie) == ([2, 3], "PIX", "PIX")     # one PCIe switch over NODE
    chosen, link, best = pick_gpus(3, [0, 1, 2, 3], pcie)
    assert link == "NODE" and same_class(link, best) and chosen[:2] in ([0, 1], [2, 3]), chosen
    # the first k GPUs of the container form the k-rank group: pairs placed first
    crossed = {0: {1: "SYS", 2: "NV4", 3: "SYS"}, 1: {0: "SYS", 2: "SYS", 3: "NV4"},
               2: {0: "NV4", 1: "SYS", 3: "SYS"}, 3: {0: "SYS", 1: "NV4", 2: "SYS"}}
    assert pick_gpus(4, [0, 1, 2, 3], crossed)[0] == [0, 2, 1, 3]


def test_gpu_pick_command_line_for_the_docker_wrapper():
    import subprocess
    from doolyprof.profiler.utils import topology
    run = lambda *a: subprocess.run([sys.executable, "-I", topology.__file__, "pick", *a],  # noqa: E731
                                    input=_TOPO_NV_PAIRS, capture_output=True, text=True)
    out = run("2", "1,2,3")
    assert out.returncode == 0 and out.stdout.split() == ["2,3", "NV12", "NV12", "1"], out
    out = run("2", "1,2")
    assert out.returncode == 0 and out.stdout.split() == ["1,2", "SYS", "NV12", "0"], out
    assert run("3", "0,1").returncode == 1
    out = subprocess.run([sys.executable, "-I", topology.__file__, "link", "1,2"],   # --devices 1,2
                         input=_TOPO_NV_PAIRS, capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.split() == ["SYS", "NV12", "0"], out


class _FakeNvml:
    """NVML of a container: ``gpus`` maps each GPU (its PCI bus id) to what its NVLink links reach
    (bus ids, or "switch"); ``visible`` are the GPUs passed into the container."""
    NVML_FEATURE_ENABLED, NVML_NVLINK_MAX_LINKS, NVML_NVLINK_DEVICE_TYPE_SWITCH = 1, 4, 2
    NVML_P2P_CAPS_INDEX_NVLINK, NVML_P2P_STATUS_OK = 3, 0

    class NVMLError(Exception):
        pass

    def __init__(self, gpus, visible, name="NVIDIA A100 80GB PCIe"):
        self.gpus, self.visible, self.name = gpus, visible, name

    def nvmlInit(self): pass
    def nvmlShutdown(self): pass
    def nvmlDeviceGetCount(self): return len(self.visible)
    def nvmlDeviceGetHandleByIndex(self, i): return self.visible[i]
    def nvmlDeviceGetName(self, h): return self.name

    def nvmlDeviceGetNvLinkState(self, h, link):
        if link >= len(self.gpus[h]):
            raise self.NVMLError("no such link")
        return 1

    def nvmlDeviceGetNvLinkRemoteDeviceType(self, h, link):
        return 2 if self.gpus[h][link] == "switch" else 0

    def nvmlDeviceGetNvLinkRemotePciInfo(self, h, link):
        import types
        return types.SimpleNamespace(busId=self.gpus[h][link])

    def nvmlDeviceGetP2PStatus(self, a, b, cap):
        both_switched = "switch" in self.gpus[a] and "switch" in self.gpus[b]
        return 0 if b in self.gpus[a] or both_switched else 5


def test_group_topology_from_the_links_of_the_group():
    # the label said "-nvlink" whenever the first GPU had an NVLink link, also for a pair
    # of GPUs that are not linked to each other and for a 4-GPU group spanning two pairs
    from doolyprof.profiler.utils import topology
    pairs = {"17": ["65", "65"], "65": ["17", "17"], "CA": ["E3", "E3"], "E3": ["CA", "CA"]}
    saved = {k: os.environ.pop(k, None) for k in ("DOOLY_TOPOLOGY", "CUDA_VISIBLE_DEVICES")}
    saved_nvml = sys.modules.get("pynvml")

    def label(gpus, visible, size, name="NVIDIA A100 80GB PCIe"):
        sys.modules["pynvml"] = _FakeNvml(gpus, visible, name)
        topology._links.cache_clear(); topology._derived_topology.cache_clear()
        return topology.group_topology(size), topology.group_link_warning(size)

    nv, pcie = "nvidia-a100-80gb-pcie-nvlink", "nvidia-a100-80gb-pcie"
    try:
        assert label(pairs, ["17", "65"], 2) == (nv, None)
        got, warning = label(pairs, ["65", "CA"], 2)             # two GPUs of different pairs
        assert got == pcie and warning and pcie in warning, (got, warning)
        assert label(pairs, ["65"], 2) == (nv, None)            # one-GPU compute container
        assert label(pairs, ["65"], 4) == (pcie, None)          # a 4-GPU group spans two pairs
        assert label(pairs, ["17", "65", "CA", "E3"], 4) == (pcie, None)
        assert label(pairs, ["17", "65", "CA", "E3"], 2) == (nv, None)   # ranks 0, 1: a pair
        assert label(pairs, ["65"], None)[0] == nv
        switched = {b: ["switch"] * 2 for b in "ABCDEFGH"}
        assert label(switched, ["C"], 8) == (nv, None)          # an NVLink switch: every GPU
        assert label(switched, list("ABCDEFGH"), 8) == (nv, None)
        rtx = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
        assert label({"1": [], "2": []}, ["1", "2"], 2, rtx)[0] == "nvidia-rtx-pro-6000-blackwell-server-edition-pcie"
        os.environ["DOOLY_TOPOLOGY"] = "mew1"
        assert label(pairs, ["65", "CA"], 2) == ("mew1", None)
    finally:
        os.environ.pop("DOOLY_TOPOLOGY", None)
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
        if saved_nvml is None:
            sys.modules.pop("pynvml", None)
        else:
            sys.modules["pynvml"] = saved_nvml
        topology._links.cache_clear(); topology._derived_topology.cache_clear()


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
