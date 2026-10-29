# STATUS

Running log of agent-driven changes to DoolyProf: what changed, why, how it was
validated, and what is still open. Newest entry first. Keep entries factual and
short; link to tests rather than repeating them. Update this file in the same
commit as the change it describes.

Conventions
- Design rule: derive logic from the engine's own definitions (vLLM / torch
  classes, type literals, constructor signatures), not hand-written tables.
- Unit tests run standalone (pytest is not installed in the dooly env):
  `python doolyprof/tests/signature_fields_test.py`,
  `python doolyprof/tests/taint_parse_test.py`.
  Real-CLI GPU check: `python doolyprof/tests/validate_signature_fields_gpu.py --workdir <dir> --gpus 0 1 2 3`.
- Env: `~/miniforge3/envs/dooly/bin/python` with
  `LD_PRELOAD=<env>/lib/libstdc++.so.6`; vLLM 0.17.1, torch 2.10, A100 80GB.

---

## 2026-10-29 — signature hash, tracer robustness, parser speed

### Signature hash = (op name, anchor MODEL_CONFIG dims, state hash, dtype)
The descendant op list is no longer hashed (`ModuleComparator._module_signature`);
it is still written to `signature_json` as a 5th element for inspection.
Why: the list described how the engine executed one instance, not what it
computes. It split identical layers through one-time work (Triton autotuning in
the first linear-attention layer, ~107k ops; per-forward chunk-index tables
cached by vLLM's `tensor_cache`; the tracer's own `PythonSubclass` dispatch
records). Over 46 databases / 1,399 signature rows it never separated two
anchors that agreed on the four hashed fields, except the attention backend,
which the state hash now carries explicitly (`impl.class` =
`type(module.impl).__qualname__`, since FlashAttentionImpl and FlashInferImpl
share one constructor contract).
Validation: old-vs-new hash audit over 41 traces (Llama 3 backends, fp8 KV,
fp16, Command R7B, Gemma-2-2b, Kimi-Linear): 184→179 and 105→104 hashes; every
absorbed hash was a KDA one-time variant whose profiled latency was within the
noise between identical layers (median +0.1..+0.4%, p5/p95 inside ±6%). All 17
assertions of `validate_signature_fields_gpu.py` pass unchanged. Attention
hashes across FLASH_ATTN / FLASHINFER / TRITON_ATTN are distinct (they differ
by 13% / up to 84% in measured latency, so merging them would be wrong).
Consequence: all hashes changed; existing databases must be re-profiled.

### State hash excludes layer-position attributes
`layer_idx`, `layer_id`, `block_idx` are excluded by name (`_NON_CONFIG_ATTRS`
in `tracer/hooks.py`). vLLM derives them from the module prefix
(`extract_layer_index`) and stores them on attention / linear-attention / MoE
layers in 34 of 251 model files; they name the instance, not its
configuration. Kimi-Linear's 20 KDA layers had produced 20 signatures.

### Tracer: registry-aware batch size, warm-up, prefix caching off
`choose_trace_batch_size` (`tracer/registry.py`) picks the trace batch `b` so
that `b`, `b+1` (cu_seqlens) and `b*prompt_len` (NUM_TOKS) avoid every value
the engine config registered as MODEL_CONFIG and the batch fits one scheduler
step. Replaces the fixed batch 7, which raised TAINT CONFLICT for models whose
config lists contain small integers (Kimi-Linear `kda_layers`: 2..27). The
tracer now runs one warm-up `generate()` on the exact prompts with taint
tracking off before the profiled one, and sets `enable_prefix_caching=False`
(vLLM only disables it for hybrid models; Llama's prefill would otherwise be
served from cache). `run-e2e.py --trace-batch-size` is the requested minimum.
Follow-up the same day: with the default 47-token prompt the batch is capped
at 11 by `max_num_batched_tokens=521`, so models that register every small
integer (Kimi-Linear 2..27, Llama-4 Scout 2..48) had no admissible batch and
the tracer raised. `choose_trace_batch` now shortens the prompt to the
longest length that admits a collision-free batch (prompt content is
irrelevant to tracing): Kimi-Linear 47 → 18 tokens x 28, Scout 48 → 10 x 51.

### Taint-string parser keeps every dimension
`profiler/taint.py` matches the tracer's label shape `LABEL(N)[hist:...]`
generically and maps labels onto `DimensionType` by value; unknown labels
(`?`, `SINGLETON`) are MODEL_CONFIG. Before, `SINGLETON(1)` was dropped, the
missing leading dim was padded as MODEL_CONFIG, and the token count leaked into
embedding and MLA signatures (504 vs 36 at different trace batches). Verified:
Kimi-Linear traced at 504 and 300 tokens gives identical signature tables.

### Parsing savings
Taint association uses a per-base-name index with a bisect time window instead
of scanning all taints for every event (`HierarchicalTraceParser._taint_index`).
The tracer keeps only `cpu_op`, `kernel`, `user_annotation` events in the
exported JSON; the parser reads nothing else.

Isolated benchmark, same input for both versions: the 270 MB Kimi-Linear-48B
trace from before the warm-up change (1,329,293 events, 5,489 taint
annotations, 244,634 events that look up taints), CPU only
(`doolyprof/tests/parse_benchmark.py`):

| stage | before (25b143a) | after | 
|---|---|---|
| JSON load | 4.7 s | 4.7 s |
| `parse_from_events` (tree, pruning, taint association) | 178.7 s | 1.8 s |
| `extract_module_to_operation_mapping` | 330.9 s | 0.8 s |
| parse + mapping | 509.6 s | 2.6 s (196x) |
| taint overlap checks | 1,342,796,026 | 5,443 |
| output | 598 hierarchies, 52 distinct | identical |

Pre-filtering the trace did not help the old parser (183.2 s + 339.0 s), so
the parse-time saving is the index alone. The export filter pays elsewhere:
file 270 MB → 98 MB, `json.dump` in the tracer 15.3 s → 5.7 s, reload in the
profiler 4.7 s → 3.0 s, about 11 s per run.

In the pipeline, Kimi-Linear-48B at fake TP 2 with a 256-token sweep (the
warm-up also removed the ~107k-op autotune burst, so the parsed trace fell
from 1.33M to 29.5k events; the module count fell with the layer-index
exclusion):

| stage | before | after |
|---|---|---|
| trace export | 126 s | 44 s |
| parse + dedup | ~8.5 min | 9 s |
| resolve | ~1.8 min | 58 s |
| profiling | 550 s (32 modules) | 83 s (13 modules) |
| end to end | 21 min 34 s | 3 min 15 s |

For Llama-3.1-8B (4 MB traces) parsing was never the bottleneck; end to end
stays at about 1 min.

### MIX dimensions follow the sweep
A MIX dimension is a product of a workload axis and a model constant (e.g. the
flattened tokens x heads tensor `MIX(8064)[hist:NUM_TOKS:504,MODEL_CONFIG:16]`).
Three defects kept its components empty or wrong: the parser read history
pairs as `qty:taint` while the tracer writes `taint:qty`; `strip('[]')` ate
the closing bracket of a history that ends an input list; and the generator
matched component labels by substring, so `MAX_NUM_REQS` buffer bounds would
have been swept as requests. Now: both pair orders parse, brackets are left
to the regex, and `input_generator._component_axis` maps labels exactly onto
the `DimensionType` values (MAX_* and MODEL_CONFIG are fixed factors). The
trace parser warns when an anchor still has a MIX dim without components.
Validation: `taint_parse_test.py` drives the real `InputGenerator` on GPU with a
synthetic `aten::mul.Tensor` over `[MIX(NUM_TOKS*16), 128]` inputs, asserts
every sweep point produces `[num_tokens*16, 128]` tensors and runs the op; a
`MAX_NUM_REQS*MODEL_CONFIG` dim stays fixed. On the Kimi-Linear and Llama
traces all 345 MIX annotations parse with components (0 before). No profiled
anchor has a MIX input in any trace so far, so databases are unaffected; Llama
re-profiled identically before/after.

### Kernel-match audit: is the state hash enough to pick a layer instance?
Resolution maps a traced class to one live submodule path (e.g.
`model.layers.0.self_attn`). After filtering candidates by state hash the
resolver runs a "kernel-match": a dummy forward of each candidate under the
torch profiler until one launches the traced kernel set. For Kimi-Linear that
single profiled forward costs 38 s (first Triton autotune in the process) and
is paid again in the profiling phase. Audit (scratch instrumentation, not in
the repo): capture kernels for EVERY state-filtered candidate and check
whether the first one already matches.

| model / backend | class | candidates after state filter | first matches | all match |
|---|---|---|---|---|
| Llama-3.1-8B / FLASHINFER | Attention | 32 of 32 | yes | 32/32 |
| Command R7B / FLASH_ATTN | Attention (global, sliding) | 8 of 32, 24 of 32 | yes | all |
| Gemma-2-2b / FLASHINFER | Attention (global, sliding) | 13 of 26, 13 of 26 | yes | all |
| gpt-oss-20b / TRITON_ATTN (sinks) | Attention (full, sliding 128) | 12 of 24, 12 of 24 | yes | all |
| gpt-oss-20b | FusedMoE | 24 | yes | 24/24 |
| Qwen3-30B-A3B / FLASHINFER | Attention | 48 | yes | 48/48 |
| Kimi-Linear / TRITON_MLA | KimiDeltaAttention | 20 | yes | 20/20 |

The only kernel-set difference among same-state candidates is a one-time
elementwise / RNG kernel on the first instance, never part of the traced set.
Where kernel-match cannot match at all (Gemma / Command R norm layers,
DeepSeek-V2-Lite and Kimi MLA, DeepSeek / Kimi MoE: dummy routing or
metadata differs from the trace) the resolver already falls back to the first
path and resolution succeeds. Conclusion: with the engine-derived state hash,
kernel-match never changed the chosen path on any audited model; it can be
skipped when all state-filtered candidates share the hash (fallback only when
the state hash is empty). Not yet implemented.

Models that could not be audited (pre-existing tracer / backend issues, not
regressions): Nemotron-51B (DeciLM) fails at model construction,
`num_attention_heads // n_heads_in_group` with `None` for its 18
replace-with-linear blocks; Llama-4 Scout fails in the tracer with
`unhashable type: numpy.ndarray`; Gemma-3-27b (multimodal) rejects
FLASHINFER and FLASH_ATTN ("partial multimodal token full attention").

### Open
- First KDA layer also carries vLLM's per-forward cached chunk tables; no
  longer visible in signatures, but that per-forward cost is not recorded
  anywhere.
- Resolution: skip kernel-match when state-filtered candidates are
  interchangeable; reuse the resolver's chosen path in the profiling phase;
  share one engine between resolution and profiling (see audit above).
- Tracer support gaps found by the audit: Nemotron-51B, Llama-4 Scout,
  Gemma-3 multimodal (details above).

---

## 2026-10-28 — engine-derived state hash, per-op dtype, kv_cache_dtype key (25b143a)
State hash reads vLLM's `CacheDType` / `MambaDType` literals, Enum attrs, the
`AttentionImpl.__init__` contract and `CacheConfig` dtypes; per-op dtype comes
from the op's traced inputs; `profiled_models` is keyed by
`(model_name, backend, tp_degree, dtype, kv_cache_dtype)`; `--kv-cache-dtype`
plumbed through tracer, resolver, profiler and the simulator.
