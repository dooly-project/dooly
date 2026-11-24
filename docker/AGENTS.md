# DoolyProf in Docker: instructions for agents

This file is for coding agents that run DoolyProf for a user. People should read
[README.md](README.md). Everything here is run from the repository root.

## What this is

- `docker/dooly-docker.sh` runs the public image `jhkjames1110/doolyprof:latest` (vLLM 0.33,
  PyTorch 2.14, CUDA 13.0, DoolyProf and DoolySim; `latest` is the repository's main branch).
- `run` traces a model (records every distinct GPU operation and how its shapes depend on the
  batch), then profiles it (times each operation over batch sizes) into a SQLite database.
- The image is **offline** (`HF_HUB_OFFLINE=1`). A model must be fetched with `download` first.
  Weights are never needed; `download` fetches only config and tokenizer files.
- Compute is always traced and profiled on one GPU, also for `--tp N` / `--dp N` (one rank
  of the split is emulated). Real multi-GPU runs are needed only for `--profile-comm`.

## Preflight (run in order; stop and report on the first failure)

| Check | Command | Expected |
|---|---|---|
| Driver | `nvidia-smi --query-gpu=index,name,driver_version,memory.used --format=csv` | GPUs listed, driver >= 580 |
| Docker access | `docker info >/dev/null && echo ok` | `ok`. If permission denied: `export DOCKER="sudo docker"` and use `$DOCKER` below (do not run `sudo` for anything else). |
| Image | `${DOCKER:-docker} pull jhkjames1110/doolyprof` | `Status: ...` with digest; 8 GB the first time |
| GPUs in containers | `${DOCKER:-docker} run --rm --gpus all --entrypoint nvidia-smi jhkjames1110/doolyprof -L` | One `GPU i: ...` line per GPU. `could not select device driver` = NVIDIA Container Toolkit missing: report to the user, do not install system packages yourself. |
| Self-test | `docker/dooly-docker.sh test` | Last line `[dooly] 12/12 suites passed`, exit 0, ~1 min |

## Commands

```bash
docker/dooly-docker.sh download MODEL [MODEL ...]                 # no GPU; seconds
docker/dooly-docker.sh run --model MODEL --out /out/NAME          # 1 GPU
docker/dooly-docker.sh --gpus N run --model MODEL --tp N --out /out/NAME --profile-comm
docker/dooly-docker.sh --devices 2,3 run ...                      # exact host GPUs, in this order
docker/dooly-docker.sh test
docker/dooly-docker.sh check-update                               # is Docker Hub's image newer?
docker/dooly-docker.sh export --db /out/NAME/profile.db --out /out/profiles   # no GPU; CSV tree
docker/dooly-docker.sh import /out/profiles --db /out/NAME/profile.db         # no GPU; merge a tree
docker/dooly-docker.sh verify /out/profiles                                   # no GPU; tree == export?
docker/dooly-docker.sh run --model MODEL --out /out/NAME --db /out/dooly.db   # profile into, reuse from, a db
docker/example.sh [dense] [moe] [hetero] [-- RUN-OPTIONS]        # example workloads, see below
```

Contributing to [DoolyDB](https://huggingface.co/datasets/dooly-project/DoolyDB) (the shared result tree on Hugging Face): `hf download
dooly-project/DoolyDB --repo-type dataset --local-dir dooly-out/DoolyDB` on the host, then
`import /out/DoolyDB --db /out/dooly.db`, `run ... --db /out/dooly.db` (reuses what DoolyDB has
for this GPU and engine), `export --db /out/dooly.db --out /out/DoolyDB`, `verify /out/DoolyDB`,
and on the host `hf upload dooly-project/DoolyDB dooly-out/DoolyDB . --repo-type dataset
--create-pr`. The pull request opens as a draft; the user publishes it on its page (the
dataset's Community tab). Uploading needs the user's `hf auth login`; never ask for or type a token.

`docker/example.sh` runs `Qwen/Qwen3.8-27B` at `--tp 2` (`FLASH_ATTN`; on 2 GPUs with
`--profile-comm` when the machine has two), `Qwen/Qwen3-30B-A3B` (`FLASHINFER`, needs an 80 GB GPU)
and `openai/gpt-oss-20b` (`TRITON_ATTN`), downloading configs it does not find in `HF_CACHE`.
None of the three is gated; no `HF_TOKEN` is needed.
It exits 0 only if every model's summary row says `ok`; results are in `$DOOLY_OUT/example/NAME/`,
each run's output in `$DOOLY_OUT/example/NAME.log`.

- `--out` must start with `/out/`; it maps to `$DOOLY_OUT` (default `./dooly-out`) on the host.
- Pass extra stage arguments with the `=` form: `--profiler-args=--overwrite`. The space form
  (`--profiler-args "--overwrite"`) is rejected by argparse.
- Other `run` options: `--dp N`, `--enable-expert-parallel`, `--attention-backend`
  (`FLASH_ATTN` default, `FLASHINFER`, `TRITON_ATTN`, MLA backends for MLA models),
  `--max-batch-size` (2), `--max-seq-len` (128), `--dtype` (`bfloat16`), `--gpu-memory-cap F`.
- Gated models: `HF_TOKEN=... docker/dooly-docker.sh download MODEL`. Never print or log the token;
  if none is available, ask the user to set `HF_TOKEN` or run `hf auth login` themselves.

## Timing (for timeouts and background jobs)

Measured on one RTX PRO 6000 Blackwell; an A100 is similar.

| Step | Typical |
|---|---|
| `download` | 1-10 s |
| `test` | ~1 min |
| `run`, small model (Qwen3-0.6B) | ~3 min first time, ~1.5 min after |
| `run`, 8B model | 3-5 min |
| `--profile-comm`, 2-4 GPUs | adds 2-8 min |
| `example.sh`, all three | ~10 min with compiled kernels (dense ~8, moe ~1.5, hetero ~1); ~15 min the first time |
| First run on a new GPU | add several minutes (kernel compilation, cached in `$DOOLY_CACHE`) |

Run anything longer than ~2 minutes in the background and poll its output.

## Success criteria

`run` prints (stderr/stdout of the script):

```
[dooly] traced in Ns: /out/NAME/trace/<model>_<tp>_dooly_trace.json
[dooly] profile exit 0 in Ns: R result rows, 0 ops without a working parent[, collective measurements: ...]
```

A run succeeded only if **all** hold:

1. the script's exit status is 0;
2. `profile exit 0` and `0 ops without a working parent`;
3. `dooly-out/NAME/profile.db` exists and `SELECT COUNT(*) FROM results` is R (> 0);
4. with `--profile-comm`: the `collective measurements:` part names a count > 0 and no
   `skipped ...-rank groups` (unless you gave fewer GPUs than ranks on purpose).

Verify with Python if `sqlite3` is not installed:

```bash
python3 - dooly-out/NAME/profile.db <<'EOF'
import sqlite3, sys
db = sqlite3.connect(sys.argv[1])
for t in ("signatures", "results", "model_operations", "coll_ops", "coll_op_results"):
    print(t, db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
EOF
```

## Outputs

`dooly-out/NAME/`: `trace/*.json` (trace), `trace.log`, `profile.db` (results), `profile.log`,
`profile_timings.csv` (profiler bookkeeping).

`profile.db` tables:

- `signatures(signature_hash, operation_name, signature_json)`: one row per distinct operation.
- `results(signature_hash, num_tokens, num_requests, is_prefill, kv_cache_size,
  prefill_chunk_size, activated_experts, local_pairs, num_tensor_parallel_workers,
  mean/median/min/max/std_latency_ms, input_shapes)`: one row per operation x batch configuration.
- `profiled_models`, `model_operations(model_id, operation_id, count)`: operations per forward pass.
- `coll_ops(operation_name, topology, tp_degree, dim_param, num_dims)`, `model_collectives`:
  the model's collectives.
- `coll_op_results(operation, topology, backend, tp_size, size_bytes, dtype, dim_param, num_dims,
  latency_ms)`: collective latencies (only with `--profile-comm`).

## Failures and what to do

| Symptom (where) | Cause | Action |
|---|---|---|
| `MODEL is not in ... (the image works offline)` (script, exit 2) | Not downloaded | `docker/dooly-docker.sh download MODEL`, then rerun |
| `LocalEntryNotFoundError` / `outgoing traffic has been disabled` (`trace.log`) | Model missing from `$HF_CACHE`, or `HF_CACHE` points elsewhere than at download time | Download again with the same `HF_CACHE` |
| `401` / `gated repo` (download), or `download failed (...)` (example.sh) | Gated model | Need `HF_TOKEN` from the user |
| `waiting for N free GPU(s)` / `waiting for GPU(s) ...` (script, repeats every 60 s) | GPUs in use by others or locked by another run (`/tmp/dooly_gpu<i>.lock`) | Wait, or tell the user. **Never** kill other processes or remove lock dirs you did not create. |
| `waiting for N free GPUs linked by ...` | Only worse-connected GPUs are free | Wait, or `DOOLY_ANY_LINK=1` if the user accepts measurements on those links |
| `trace failed (exit N)` (script) + vLLM error about the attention backend or head size (`trace.log`) | vLLM does not support that backend on this GPU/model | Retry with another `--attention-backend`; report which ones vLLM rejected |
| KV cache cannot hold the trace batch (`trace.log`/`profile.log`) | Not enough memory for the batch | `--gpu-memory-cap 0.95` |
| CUDA out of memory | Model rank too big for the GPU | Larger `--tp` (still one GPU), or a larger GPU |
| `ops without a working parent` > 0 or `profile exit 1` | Profiler could not run some operation | Keep the run folder; report `profile.log` to the user (lines with `no working parent`) |
| `argument --profiler-args: expected one argument` | Space form | Use `--profiler-args=...` |
| `profile exit 0 ... 0 result rows` on a rerun | Same `--out`: everything already in its `profile.db` | New `--out`, or `--profiler-args=--overwrite` |
| `[WARN] Calls recorded on K-rank groups are not profiled` (`profile.log`) | Fewer GPUs than group size | Rerun with `--gpus K` if the user wants those measured |
| `[WARN] the K GPUs of this group are not all linked by NVLink` (`profile.log`) | Run got GPUs from different NVLink islands | Results are stored under the `-pcie` label; rerun without `--devices`/`DOOLY_ANY_LINK` for NVLink numbers |

## Rules

- Do not run on GPUs other users are using; the script checks this, so do not bypass it with raw
  `docker run --gpus`.
- Do not download weights, edit the image, or install system packages; report missing
  prerequisites to the user.
- Do not use `pkill -f` with broad patterns over SSH; it can match your own session.
- A run into an existing `--out` folder reuses its `profile.db`: operations already in it are not
  profiled again and the summary reports only the new rows (possibly `0 result rows`). Use a new
  `--out` per run, or `--profiler-args=--overwrite` to measure again. `example.sh` clears each
  example's folder itself.
- Before each command the script warns if Docker Hub has a newer image than the machine
  (`Docker Hub has a newer ...`). It still runs the local copy; tell the user, and pull only if
  they agree (`DOOLY_UPDATE_CHECK=0` silences it).
- Results from different GPUs or link types are kept apart by their topology label; do not
  merge databases from different machines unless the user asks.

## Maintainers: publishing a new image

The image is the full build (`docker/Dockerfile`) of a commit of main: its Python packages
(`docker/requirements.txt`) and the `doolyprof/` and `doolysim/` packages, with the commit in the
label `org.opencontainers.image.revision`. Rebuild and push it when main changes (needs `docker
login` as the image owner; an agent must not enter credentials), from a clean copy of the commit:

```bash
SHA=$(git rev-parse main); SRC=$(mktemp -d)
git archive "$SHA" docker doolyprof doolysim | tar -x -C "$SRC"
docker build -f "$SRC/docker/Dockerfile" --label org.opencontainers.image.revision=$SHA \
  -t jhkjames1110/doolyprof:latest "$SRC"
DOOLY_IMAGE=jhkjames1110/doolyprof:latest docker/dooly-docker.sh test     # every suite must pass
docker push jhkjames1110/doolyprof:latest
```

When only `doolyprof/` or `doolysim/` changed, a thin image on top of the published one builds in
seconds: `FROM jhkjames1110/doolyprof:latest`, `RUN rm -rf /opt/dooly/doolyprof
/opt/dooly/doolysim`, `COPY` both from the archive, and the same label.
