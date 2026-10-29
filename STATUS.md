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
  vLLM 0.32.0: `~/miniforge3/envs/dooly-v32/bin/python` (no LD_PRELOAD), torch 2.13.
  `python doolyprof/tests/tracer_ops_test.py` covers tracer op wrappers.

---

## 2026-10-29 — vLLM 0.32.0 port
Env: new conda env `dooly-v32` (vLLM 0.32.0, torch 2.13.0+cu130, transformers
5.19, flashinfer 0.7.0.post1). `dooly` (0.17.1) is untouched and DoolySim still
uses it. No `LD_PRELOAD` needed in `dooly-v32` (sqlite loads fine after torch).
The 0.32 wheel registers `KimiK3ForConditionalGeneration` and ships FA2 (sm80).
All changes below keep 0.17.1 working: each API difference is picked from the
engine's own signature / attributes, never from a version string.

### What 0.32 broke and the fix
| # | Symptom on 0.32 | Cause | Fix |
|---|---|---|---|
| 1 | `import vllm._C` fails | ops now in `_C_stable_libtorch` / `_moe_C_stable_libtorch` | `current_platform.import_kernels()` (importer.py) |
| 2 | resolver/profiler: `set_forward_context() got unexpected 'virtual_engine'` | pipeline virtual engines removed | `utils/vllm_compat.py`: drop optional kwargs the callee does not declare |
| 3 | `BlockTable.__init__() got unexpected 'kernel_block_size'` | alloc block = kernel block | same helper (`make_block_table`) |
| 4 | `compute_slot_mapping() missing 'positions'` -> no attention metadata (MLA wrapper unresolvable) | slot mapping now a GPU kernel over `(num_reqs, query_start_loc, positions)` reading the committed device table | `fill_slot_mapping`, contract chosen by parameter names |
| 5 | first decoder layer untainted (layer-0 norm/QKV missing, 31 of 32 layers) | `VocabParallelEmbedding.forward` ends with `output.view(output_dtype)`; tracer's `view(dtype)` returned a plain tensor | keep taints through `view(dtype)` (last dim untainted if itemsize changes) |
| 6 | Kimi / Qwen3.5 tracer crash `'<' not supported between tuple and int` | Mamba `bind_kv_cache` calls `squeeze(dim=(1, 2))` | tuple dims in the squeeze wrapper |
| 7 | Kimi dedup crash `unhashable type: 'list'` | KDA metadata `torch.cat([zero, cu_seqlens])`: Kineto records a TensorList as one input holding a list of shapes, the taint string has one segment per tensor; finput paired them 1:1, misaligning every later argument | `ProfileInput` type `TensorList` with `elements`; finput consumes one taint segment per element; generator builds a list; signature params `input{i}_t{k}_dim{j}` |
| 8 | MoE profiled 1-D (DSv2 35 rows instead of 168) | `FusedMoE` class replaced by `FusedMoEFactory` -> `MoERunner`; router `_get_indices_type` gone, `_compute_routing` gains `input_ids` | `moe_hook` finds the MoE layer by its `FusedMoEConfig` (`moe_config`) + `router`; top-k / experts from that config; forced routing takes the caller's `indices_type` |
| 9 | Kimi KDA and Qwen3.5 full-attention compute absent from the profile (274 orphan kernels) | KDA / GDN / gated-norm / fused qk-norm-rope kernels are Triton driver launches outside any aten op; Kineto gives them no `External id` | `tracer/kineto_links.py` rebuilds the link from Kineto's `correlation` (innermost CPU event enclosing the launch) before the export filter drops launch events; parser indexes MODULE/COMPILE events and anchors a kernel whose direct parent is a MODULE to that module |
| 10 | sampling cost missing from every 0.32 profile | V2 runner (0.32 default) samples with a plain `Sampler` class (Triton Gumbel / block argmax), not an `nn.Module`: no MODULE scope, and the object is on the runner, not in the model | tracer wraps `__call__` of non-Module runner components (`hooks.ENGINE_CALLABLE_ATTRS = ("sampler",)`) in the usual MODULE annotation; resolver falls back to the runner when a MODULE is not in the model and profiles it through `vllm_compat.EngineSamplerCall`, which builds the batch with the runner's own `InputBatch.make_dummy` (as `_dummy_sampler_run` does) |

Not bugs, recorded: the `Invalid process group specified` traceback at engine
shutdown is harmless (fake TP group teardown); `_C::vocab_parallel_embedding`
replaces the embedding module entry (one fused kernel now).

### Validation (5 models, 256 tok / batch 2; Llama & Kimi at fake TP 2)
- Sampler (fix 10): every model on 0.32 / V2 gains one `Sampler` signature
  (2 rows, swept over requests; Llama 0.033 / 0.040 ms at 1 / 2 requests vs
  0.04 ms for 0.17's argmax + to.dtype). On 0.17 and on 0.32 with
  `VLLM_USE_V2_MODEL_RUNNER=0` the sampler is an nn.Module, nothing is wrapped,
  and the V1 argmax / to.dtype rows are profiled as before (Llama 491 rows on
  both, identical apart from the fused embedding op).
- 0.32 vs 0.17 baseline: Llama 491 -> 485 rows before fix 10;
  DSv2 / gpt-oss MoE rows equal (168 / 134); Kimi MoE 195 = 195, KDA module
  168 = 168 (now `KimiGatedDeltaNetAttention`), MLA via `MultiHeadLatentAttention`;
  Qwen3.5 GDN layer now profiled (168 rows; 0 rows on 0.17).
- 0.17 rerun with the new code vs the original HEAD baseline: identical on all
  five models except Qwen3.5 gaining `MRotaryEmbedding` (its Triton mrope
  kernel was an orphan before, fix 9) and Kimi's KDA gate kernel folding into
  the KDA module (same row count).
- Unit tests (both envs): signature_fields 12, taint_parse 16, tracer_ops 7
  (new file: view(dtype), squeeze tuple dims, kernel link repair, engine
  callable annotation); the tracer_ops tests fail on the unfixed
  code.

### Open (0.32)
- Sampler profile covers the greedy path only (the tracer's temperature-0
  requests; dummy slots default to temperature 0). Random / top-k / top-p /
  penalties / logprobs paths cost more and are not swept.
- GemmaRMSNorm (Qwen3.5) runs vLLM IR's native decomposition eagerly (fp32
  weight fails `vllm_c`'s dtype check) and is profiled as separate
  pow/mean/rsqrt/mul ops instead of one compiled module.
- Raw-op dummy runs still fail on `aten::pad` (mode/value order),
  `_vllm_fa2_C::varlen_fwd` (optional tensor args) and
  `vllm::moe_forward_shared` (`input_ids`); all recover by walking up.
- Fix 9 can make a whole module the anchor (Qwen3NextAttention now profiled as
  one unit incl. its projections), coarser than before.
- DoolySim on 0.32: its emulator stubs `GPUModelRunner.profile_run` of the V1
  runner only; with the V2 default it needs `VLLM_USE_V2_MODEL_RUNNER=0` or the
  same stub on `vllm.v1.worker.gpu.model_runner`. Not changed.
- Kimi-K3 still needs Blackwell for MXFP4 kernels and its size (~1.56 TB) far
  exceeds one A100 shard even with EP; layer truncation (`hf_overrides`) is the
  path to try next on this machine.

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
FLASHINFER and FLASH_ATTN ("partial multimodal token full attention") and on
TRITON_ATTN the traced forward fails with a broadcast error
(`[336] vs [336, 5376]`) under the taint hooks.

### Kernel names retired (kernel-match and trace dedup)
Removed the resolver's kernel-match (`_pick_path_by_kernels`,
`_capture_kernels`, path cache), the kernel-name tuples in the trace dedup
key, the parser's `descendant_kernels` annotation and the dead
`_get_*kernel_signature` helpers. Instance choice is now: state-hash filter,
then the ancestor-class filter for classes without configuration
(GemmaRMSNorm, LayerNorm). `_resolved_path` is recorded after the final
choice, and the resolver warns when no instance has the traced state hash.
History: kernel-match (Apr 20) predates the state hash (Apr 25), which was
added in front of it for FA2/Triton sliding windows and made it redundant.
Validation: all 17 GPU-script assertions pass; signatures and per-model op
counts are identical to the kernel-match databases for Llama (auto, fp8 KV,
fp16), Llama + Command R7B, Gemma-2-2b, Llama on TRITON_ATTN and Kimi-Linear;
norm instances picked are unchanged (`model.layers.0.input_layernorm`).
Kimi-Linear resolution 58 s → 31 s, end to end 3 min 15 s → 2 min 19 s.
The new warning surfaced a pre-existing mismatch: Kimi-Linear's traced
`MLAAttention` state hash (3f2c005075bb) matches none of its 7 instances at
resolve time (kernel-match also failed there, so the first instance was
already used; the 7 layers are configured identically).

### State hash: compile anchors, MLA workspace size
- Compile anchors (modules whose forward runs a `torch.compile`d function,
  e.g. GemmaRMSNorm, Command R LayerNorm) now take the state hash of the
  containing MODULE event, like their name, inputs and taint string already
  did; before, they read the COMPILE event's args, which never carry a hash,
  so these modules had an empty state hash at resolution.
- `chunked_prefill_workspace_size` excluded: vLLM sizes this MLA scratch buffer
  from engine capacity (`max_num_seqs x block_size`, `max_model_len`); the
  tracer engine (max_num_seqs 263) and the resolver engine (256) differ, so
  the traced hash matched no instance (Kimi-Linear 504,960 vs 491,520,
  DeepSeek-V2-Lite 65,536 vs 16,384).
Validation: all GPU-script assertions pass; no resolver state-mismatch
warnings on Llama, Command R7B, Gemma-2, Kimi-Linear, DeepSeek-V2-Lite,
Qwen3.5-4B. Only the compile-anchor norm signatures changed hash
(Gemma-2 / Qwen3.5 GemmaRMSNorm, Command R LayerNorm), with unchanged counts;
all other signatures and per-model op counts identical.

### Resolver uses the instance path recorded at trace time
The tracer maps every module to its dotted path once after load
(`hooks.register_module_paths`) and writes `PATH:<path>` as the last field of
each MODULE annotation. The parser stores it in the event args, trace dedup
keeps every member's path (`ModuleInfo.instance_paths`, never hashed), and the
resolver fetches the instance with `get_submodule(path)` instead of inferring
it from class name, state hash, ancestor class and module order. Before using
a path it checks the dedup group: every member must exist, carry the traced
state hash, and have the same fingerprint (state hash + name/shape of every
parameter in its subtree); otherwise it prints an ERROR naming the differing
parameters (a missing signature field). The old filters remain only for traces
without `PATH:`.
Validation: all 17 GPU-script assertions pass; signatures and per-model op
counts identical to the previous run on Llama (auto/fp8/fp16), Llama +
Command R7B, Gemma-2-2b, Kimi-Linear, DeepSeek-V2-Lite, Qwen3.5-4B and
gpt-oss-20b; zero group-check errors on all of them. Qwen3.5's query norm now
resolves directly to `layers.3.self_attn.q_norm` (before: wrong
`input_layernorm` first, recovered only by a failed dummy run).

### Open
- Other module attributes vLLM derives from deployment / engine capacity,
  not seen in the 8 models dumped so far but would enter the state hash:
  `sparse_attn_indexer.max_model_len` (DeepSeek V3.2 sparse attention),
  `static_sink_attention.block_size` / `num_sink_blocks` (cache block size),
  Phi-3 long-RoPE `use_long_rope` (= max_model_len > original context; this
  one changes computation, so it may belong in the deployment key), and the
  `scheduler_config` objects stored by Mamba-family models (not hashed today
  because not primitive). A general rule (values computed from scheduler /
  cache capacity are not configuration) would replace name-by-name exclusion.
- State hash ignores parameter shapes: Qwen3.5 GemmaRMSNorm with weight 256
  (q_norm / k_norm) and 2560 (layer norms) share a hash; RowParallelLinear
  with and without bias share a hash.
- Ancestor-class filter (still used for traces without `PATH:`) is
  order-dependent and, for compile anchors, sees the module itself as the
  first ancestor; superseded by recorded paths for new traces.
- A resolver group-check ERROR (members of one dedup group with different
  configuration or parameter shapes) currently only reports; splitting the
  group would need a new signature field. None observed on 8 models.
- First KDA layer also carries vLLM's per-forward cached chunk tables; no
  longer visible in signatures, but that per-forward cost is not recorded
  anywhere.
- Resolution: reuse the resolver's chosen path in the profiling phase; share
  one engine between resolution and profiling.
- Tracer support gaps found by the audit: Nemotron-51B, Llama-4 Scout,
  Gemma-3 multimodal (details above).

---

## 2026-10-28 — engine-derived state hash, per-op dtype, kv_cache_dtype key (25b143a)
State hash reads vLLM's `CacheDType` / `MambaDType` literals, Enum attrs, the
`AttentionImpl.__init__` contract and `CacheConfig` dtypes; per-op dtype comes
from the op's traced inputs; `profiled_models` is keyed by
`(model_name, backend, tp_degree, dtype, kv_cache_dtype)`; `--kv-cache-dtype`
plumbed through tracer, resolver, profiler and the simulator.
