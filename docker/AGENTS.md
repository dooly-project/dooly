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
- Every command of `dooly-docker.sh` and `example.sh` needs a Hugging Face token:
  `export HF_TOKEN_FILE=/path/to/token.txt`, the path of a file that holds it (ask the user for
  the path), or `HF_TOKEN`. Only `download`, `download-db` and `upload` hand it to a container,
  and they never use a login saved on the machine.
- On an account other people use too (e.g. a shared `ubuntu`), first point `HF_CACHE` and
  `DOOLY_CACHE` at folders of your own (e.g. in your workspace): the defaults,
  `~/.cache/huggingface` and `~/.cache/dooly-docker`, are that account's. Keep the same `HF_CACHE`
  from `download` to `run`. There the script cannot tell other people's GPU processes from yours:
  it goes by memory in use, utilization and the lock folders ([Rules](#rules)).

## Tested GPUs, and what to do on others

DoolyProf has been tested only on NVIDIA **RTX PRO 6000 Blackwell Server Edition** (96 GB, compute
capability 12.0) and **A100 80GB PCIe** (8.0) GPUs, with the image's vLLM, PyTorch and CUDA. On any
other GPU expect bugs nobody has hit yet, typically where vLLM picks a different implementation for
that GPU: FlashAttention 4 or a newer FlashInfer path on other Blackwell GPUs, other Triton or
CUTLASS kernels, or kernels compiled on first use that fail for that compute capability. These bugs
are yours to deal with; do not hand them back to the user as they are:

1. Find the cause in `trace.log`, `profile.log` and the traceback: which operation, which library,
   which call.
2. Try what needs no code change first: another `--attention-backend` that vLLM supports on this GPU,
   `--gpu-memory-cap`, or an empty `DOOLY_CACHE` of your own if a compiled kernel is broken.
3. If DoolyProf itself is wrong (the tracer misses or misrecords an operation, the profiler cannot
   run one, `ops without a working parent` > 0), fix it in your clone of the repository. The fix
   must be general: no special case for one GPU, library version, model or configuration (no
   hard-coded names, versions or shapes), and nothing other GPUs measure may change. Commit it in
   your clone, build a thin local image with it (the recipe at the end of this file, tagged e.g.
   `doolyprof:local-fix`, with `--build-arg DOOLY_REVISION=$(git rev-parse HEAD)`) and run every
   command with `DOOLY_IMAGE=doolyprof:local-fix`. Never push that image.
4. With a fix, `docker/dooly-docker.sh test` must still pass every suite and the run must meet the
   success criteria below.
5. Tell the user what failed, why, and what you changed (the diff or the commit in your clone), so
   the fix can go into the repository. If you contribute to DoolyDB with it, say so in your report:
   the pull request stays a draft until the user has looked.

## Preflight (run in order; stop and report on the first failure)

| Check | Command | Expected |
|---|---|---|
| Token | `test -r "$HF_TOKEN_FILE" && test -s "$HF_TOKEN_FILE" && echo ok` | `ok`. Otherwise ask the user for the path of their token file. Never read, print or copy it. |
| Driver | `nvidia-smi --query-gpu=index,name,driver_version,memory.used --format=csv` | GPUs listed, driver >= 580 |
| Docker access | `docker info >/dev/null && echo ok` | `ok`. If permission denied: `export DOCKER="sudo docker"` and use `$DOCKER` below (do not run `sudo` for anything else). |
| Image | `${DOCKER:-docker} pull jhkjames1110/doolyprof` | `Status: ...` with digest; 8 GB the first time. Skip it if the user says the image here is the one to use; `docker/dooly-docker.sh check-update` then says whether it or Docker Hub's was built later. |
| GPUs in containers | `${DOCKER:-docker} run --rm --gpus all --entrypoint nvidia-smi jhkjames1110/doolyprof -L` | One `GPU i: ...` line per GPU. `could not select device driver` = NVIDIA Container Toolkit missing: report to the user, do not install system packages yourself. |
| Self-test | `docker/dooly-docker.sh test` | Last line `[dooly] N/N suites passed` (every suite passed), exit 0, ~1 min |

## Commands

```bash
docker/dooly-docker.sh download MODEL [MODEL ...]                 # no GPU; seconds
docker/dooly-docker.sh run --model MODEL --out /out/NAME          # 1 GPU
docker/dooly-docker.sh --gpus N run --model MODEL --tp N --out /out/NAME --profile-comm
docker/dooly-docker.sh --devices 2,3 run ...                      # exact host GPUs, in this order
docker/dooly-docker.sh test
docker/dooly-docker.sh check-update                               # is Docker Hub's image built later?
docker/dooly-docker.sh export --db /out/NAME/profile.db --out /out/profiles   # no GPU; CSV tree
docker/dooly-docker.sh import /out/profiles --db /out/NAME/profile.db         # no GPU; merge a tree
docker/dooly-docker.sh verify /out/profiles                                   # no GPU; tree == export?
docker/dooly-docker.sh run --model MODEL --out /out/NAME --db /out/dooly.db   # profile into, reuse from, a db
docker/dooly-docker.sh download-db /out/DoolyDB [--revision refs/pr/N]        # no GPU; DoolyDB, or a PR's revision
docker/dooly-docker.sh upload /out/DoolyDB "ORG/MODEL on GPU"                 # no GPU; verify, then a draft PR
docker/example.sh [dense] [moe] [hetero] [-- RUN-OPTIONS]        # example workloads, see below
```

Contributing to DoolyDB: see [its section](#contributing-to-doolydb) below.

`docker/example.sh` runs `Qwen/Qwen3.8-27B` at `--tp 2` (`FLASH_ATTN`; on 2 GPUs with
`--profile-comm` when the machine has two), `Qwen/Qwen3-30B-A3B` (`FLASHINFER`, needs an 80 GB GPU)
and `openai/gpt-oss-20b` (`TRITON_ATTN`), downloading configs it does not find in `HF_CACHE`.
None of the three is gated, but like every command it needs the token (`HF_TOKEN_FILE`).
It exits 0 only if every model's summary row says `ok`; results are in `$DOOLY_OUT/example/NAME/`,
each run's output in `$DOOLY_OUT/example/NAME.log`.

- `--out` must start with `/out/`; it maps to `$DOOLY_OUT` (default `./dooly-out`) on the host.
- Pass extra stage arguments with the `=` form: `--profiler-args=--overwrite`. The space form
  (`--profiler-args "--overwrite"`) is rejected by argparse.
- Other `run` options: `--dp N`, `--enable-expert-parallel`, `--attention-backend`
  (`FLASH_ATTN` default, `FLASHINFER`, `TRITON_ATTN`, MLA backends for MLA models),
  `--max-batch-size` (2), `--max-seq-len` (128), `--dtype` (`bfloat16`), `--gpu-memory-cap F`.
- Gated models: the token's account must have accepted the model's license on its page.

## Contributing to DoolyDB

[DoolyDB](https://huggingface.co/datasets/dooly-project/DoolyDB) is the shared result tree on
Hugging Face. You do the whole contribution, upload included, and you check it yourself:
`dooly verify` only proves the folder is what `dooly export` writes, not that it holds what you
meant to add. The walkthrough for people is docs/how-to-contribute.md.

```bash
export HF_TOKEN_FILE=/path/to/token.txt                     # from the user
docker/dooly-docker.sh download-db /out/DoolyDB-base        # untouched copy, to compare against
docker/dooly-docker.sh download-db /out/DoolyDB             # the copy your results go into
docker/dooly-docker.sh import /out/DoolyDB --db /out/dooly.db
docker/dooly-docker.sh download MODEL
docker/dooly-docker.sh run --model MODEL --out /out/NAME --db /out/dooly.db --max-batch-size 32 --max-seq-len 8192
docker/dooly-docker.sh export --db /out/dooly.db --out /out/DoolyDB
docker/dooly-docker.sh verify /out/DoolyDB
# your own checks (below), then:
docker/dooly-docker.sh upload /out/DoolyDB "MODEL on GPU"
# your checks of the pull request (below)
```

`upload` runs `dooly verify` again and stops if it fails, uploads as the token's account (it
prints which), always as a pull request (never a commit to main), and prints the pull request's
number and URL and what it adds, changes and removes compared with main.

`download-db` leaves Hugging Face's download bookkeeping in `.cache/huggingface/` inside the
folder; `verify` and `upload` ignore it (hence `--exclude=.cache` below). `hf` may also leave its own
`.agent_harnesses.json` in `HF_CACHE`; it is harmless. While DoolyDB has nothing
for this GPU and engine version, `import` adds 0 rows and the run measures every operation; later
runs reuse what is there (a model's variants share most operations).

Check before uploading (stop and report on any failure; do not upload a folder you could not check):

1. **Only your folder is new.** `diff -rq --exclude=.cache dooly-out/DoolyDB-base dooly-out/DoolyDB`
   prints only `Only in dooly-out/DoolyDB/...: X` lines, where X is your deployment's folder or the
   highest folder above it that DoolyDB did not have yet (in an empty DoolyDB, `Only in
   dooly-out/DoolyDB: compute`); with `--profile-comm` also the collective files of your topology
   under `communication/`. When X is above your deployment, `find dooly-out/DoolyDB/compute -name
   deployment.json` must list only yours besides those of the base. A `Files ... differ` line for
   an existing deployment means stop.
2. **The folder names what you ran**: `compute/<gpu>/<engine>/<backend>/<org>/<model>/<deployment>/`,
   with `<gpu>` this GPU (`nvidia-smi` name), `<engine>` the image's vLLM version, `<backend>` your
   `--attention-backend`, and `<deployment>` your `tp<N>-<dtype>-kv<cache dtype>`.
3. **deployment.json** has the right `model`, `device`, `engine`, `backend` and `dtype`, and the
   command line in `runs[].stack.argv` holds no path you would not share (paths under `/out/` are
   fine). The same goes for your run's logs and trace in the folder's `runs/<run_uid>/` (`run_uid`
   from `runs[]`): `zcat .../runs/<run_uid>/*.gz | grep -nE "/home/|$(whoami)|$(hostname)"` should
   print nothing you would not share. DoolyProf writes home directories as `~`, and your user and
   host names as `<user>` and `<host>`; in Docker the logs only show paths inside the container.
4. **model.csv lists every operation file and nothing is missing**: each `file` it names exists,
   and every other CSV in the folder is `model.csv` or `collectives.csv`; `runs/<run_uid>/` of your run
   holds `trace.log.gz`, `profile.log.gz` and `trace.json.gz`. Per-layer operations have a `count`
   equal to the model's `num_hidden_layers` (its `config.json` in `HF_CACHE`) or a multiple of it;
   the embedding and the sampler run once.
5. **The numbers are sane**: every operation file has rows; all latencies are positive and finite,
   and min <= median <= max on each row; a linear operation's median latency is higher at the
   largest `num_tokens` than at the smallest. Between neighbouring sizes it may dip (the matrix
   library switches kernels) or stay flat (the output layer, whose large weight dominates small
   batches): that is normal. A curve that falls overall, or the same latency at every size, is not.
   The folder's rows add up to R from the run's summary when the database had none of these
   operations before.

Check after uploading:

6. `upload` printed `pull request #N`, the account you expected, `changes 0, removes 0` (unless
   step 1 showed collective files) and only your folder after `+`.
7. **The pull request holds exactly your folder**:
   `docker/dooly-docker.sh download-db /out/pr-N --revision refs/pr/N`, then
   `diff -rq --exclude=.cache dooly-out/DoolyDB dooly-out/pr-N` prints nothing, and
   `docker/dooly-docker.sh verify /out/pr-N` is valid.
8. **It is an open draft by that account** (public dataset, no token needed):
   `${DOCKER:-docker} run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HUB_OFFLINE=0 --entrypoint python jhkjames1110/doolyprof -c
   "from huggingface_hub import HfApi; d = HfApi().get_discussion_details('dooly-project/DoolyDB', N, repo_type='dataset'); print(d.status, d.author, d.title)"`
   prints `draft <account> <your message>`.

Then report the pull request's URL and the result of each check. It stays a draft: the user
publishes it on its page (the dataset's Community tab) unless they told you to. If a check fails
after uploading, do not edit the pull request; tell the user (fix the folder and upload a new one
if they agree).

Token rules: use only the token file the user gave you (or their `HF_TOKEN`). Never read, print,
copy or log it, never put a token on a command line or in a file you write, never ask for one in
chat, and never use a login saved on the machine or a token you find elsewhere.

## Timing (for timeouts and background jobs)

Measured on one RTX PRO 6000 Blackwell; an A100 is similar.

| Step | Typical |
|---|---|
| `download` | 1-10 s |
| `download-db`, `upload` | seconds to a minute (grows with DoolyDB) |
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
[dooly] kept trace.log, profile.log and the trace in /out/... with run <run_uid> (dooly export writes them ...)
```

A run succeeded only if **all** hold:

1. the script's exit status is 0;
2. `profile exit 0` and `0 ops without a working parent`;
3. `dooly-out/NAME/profile.db` exists and has the run's rows:
   `SELECT COUNT(*) FROM results WHERE run_id = (SELECT MAX(run_id) FROM profile_runs)` is R (> 0).
   R counts the rows this run added: a batch shape measured twice in one run is one row;
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

`dooly-out/NAME/`: `trace/*.json` (trace), `trace.log`, `profile.log`, and `profile.db`
(results) with `profile_timings.csv` (profiler bookkeeping, how long each operation took to
measure). With `--db /out/FILE.db` the results go into that file and `profile_timings.csv` next to
it, and the run folder has no `profile.db`.

The database also keeps the run's `trace.log`, `profile.log` and trace, gzipped (table `run_files`,
the `kept ...` line of `run`), and `export` writes them into the deployment's folder as
`runs/<run_uid>/trace.log.gz`, `profile.log.gz` and `trace.json.gz` (for collectives also under
`communication/.../runs/`). They go to DoolyDB with your measurements, so a run can be debugged
later from DoolyDB alone: `zcat .../runs/<run_uid>/profile.log.gz | less`. A run that fails before
the profiler records it keeps its logs only in `dooly-out/NAME/`.

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
- `profile_runs(run_uid, device, engine, backend, stack_json)`, `run_files(run_uid, name, data)`:
  each run's versions and command line, and its gzipped logs and trace.

## Failures and what to do

| Symptom (where) | Cause | Action |
|---|---|---|
| `MODEL is not in ... (the image works offline)` (script, exit 2) | Not downloaded | `docker/dooly-docker.sh download MODEL`, then rerun |
| `LocalEntryNotFoundError` / `outgoing traffic has been disabled` (`trace.log`) | Model missing from `$HF_CACHE`, or `HF_CACHE` points elsewhere than at download time | Download again with the same `HF_CACHE` |
| `a Hugging Face token is required` / `HF_TOKEN_FILE=... is not a readable, non-empty file` (script, exit 2) | No token given, or a wrong path | Ask the user for the path of their token file |
| `401` / `gated repo` (download), `download failed (...)` (example.sh), `the Hugging Face token was not accepted` (upload) | Token invalid or expired, or a gated model whose license the account has not accepted | Tell the user; do not look for another token |
| `not uploaded: dooly verify found problems` (upload) | The folder is not what `dooly export` writes | docs/how-to-contribute.md, *If verify fails* |
| `waiting for N free GPU(s)` / `waiting for GPU(s) ...` (script, repeats every 60 s) | GPUs in use (4 GB or more used, kernels running, or another user's process) or locked by another run (`/tmp/dooly_gpu<i>.lock`) | Wait, or tell the user. **Never** kill other processes or remove lock dirs you did not create. |
| `waiting for N free GPUs linked by ...` | Only worse-connected GPUs are free | Wait, or `DOOLY_ANY_LINK=1` if the user accepts measurements on those links |
| `trace failed (exit N)` (script) + vLLM error about the attention backend or head size (`trace.log`) | vLLM does not support that backend on this GPU/model | Retry with another `--attention-backend`; report which ones vLLM rejected |
| KV cache cannot hold the trace batch (`trace.log`/`profile.log`) | Not enough memory for the batch | `--gpu-memory-cap 0.95` |
| CUDA out of memory | Model rank too big for the GPU | Larger `--tp` (still one GPU), or a larger GPU |
| `ops without a working parent` > 0 or `profile exit 1` | Profiler could not run some operation | Keep the run folder; report `profile.log` to the user (lines with `no working parent`) |
| `argument --profiler-args: expected one argument` | Space form | Use `--profiler-args=...` |
| `profile exit 0 ... 0 result rows` on a rerun | Same `--out`: everything already in its `profile.db` | New `--out`, or `--profiler-args=--overwrite` |
| `[WARN] Calls recorded on K-rank groups are not profiled` (`profile.log`) | Fewer GPUs than group size | Rerun with `--gpus K` if the user wants those measured |
| `[WARN] the K GPUs of this group are not all linked by NVLink` (`profile.log`) | Run got GPUs from different NVLink islands | Results are stored under the `-pcie` label; rerun without `--devices`/`DOOLY_ANY_LINK` for NVLink numbers |
| `Chat template warmup failed` with a traceback, `FlashInfer's precompiled kernels are missing`, `Unknown vLLM environment variable detected: VLLM_FAKE_*` (`trace.log`, `profile.log`) | Harmless: vLLM runs without a tokenizer, compiles the FlashInfer kernels it needs into `$DOOLY_CACHE`, and does not know DoolyProf's own `VLLM_FAKE_*` settings (rank emulation) | Nothing; judge the run by the success criteria |
| `Wrote N results` (`profile.log`) above R of the summary | A batch shape measured twice in one run is stored once | Nothing; R is what the database got |

## Rules

- Do not run on GPUs other users are using; the script checks this (a GPU is free with under 4 GB
  in use, under 5% utilization, no other user's process and no lock), so do not bypass it with raw
  `docker run --gpus`.
- Do not download weights or install system packages; report missing prerequisites to the user.
  Do not change or push the published image; a local image with your own fix is fine (see
  [Tested GPUs](#tested-gpus-and-what-to-do-on-others)).
- Do not use `pkill -f` with broad patterns over SSH; it can match your own session.
- A run into an existing `--out` folder reuses its `profile.db`: operations already in it are not
  profiled again and the summary reports only the new rows (possibly `0 result rows`). Use a new
  `--out` per run, or `--profiler-args=--overwrite` to measure again. `example.sh` clears each
  example's folder itself.
- Before each command the script warns if Docker Hub's image was built later than the machine's
  (`Docker Hub has a newer ... (built DATE from commit C) than this machine (built ...)`). It still
  runs the local copy; tell the user, and pull only if they agree (`DOOLY_UPDATE_CHECK=0` silences
  it). `check-update` says `... is not older than Docker Hub's ...: nothing to update` when the
  copy here was built later (e.g. from a newer commit of main), and `... are builds of the same
  commit: nothing to update` when both come from one commit; neither warns before other commands.
- Results from different GPUs or link types are kept apart by their topology label; do not
  merge databases from different machines unless the user asks.

## Maintainers: publishing a new image

The image is the full build (`docker/Dockerfile`) of a commit of main: its Python packages
(`docker/requirements.txt`) and the `doolyprof/` and `doolysim/` packages, with the commit in the
label `org.opencontainers.image.revision` and in `/opt/dooly/doolyprof/REVISION` (which every
profiling run records as `doolyprof_revision`), both from `--build-arg DOOLY_REVISION`. Rebuild and
push it when main changes (needs `docker login` as the image owner; an agent must not enter
credentials), from a clean copy of the commit:

```bash
SHA=$(git rev-parse main); SRC=$(mktemp -d)
git archive "$SHA" docker doolyprof doolysim | tar -x -C "$SRC"
docker build -f "$SRC/docker/Dockerfile" --build-arg DOOLY_REVISION=$SHA \
  -t jhkjames1110/doolyprof:latest "$SRC"
DOOLY_IMAGE=jhkjames1110/doolyprof:latest docker/dooly-docker.sh test     # every suite must pass
docker push jhkjames1110/doolyprof:latest
```

When only `doolyprof/` or `doolysim/` changed, the machine that built the previous image reuses
its cached layers up to `COPY doolyprof`, so the build takes seconds and the push uploads only the
code layers. Elsewhere, a thin image on top of the published one does the same:
`FROM jhkjames1110/doolyprof:latest`, `RUN rm -rf /opt/dooly/doolyprof /opt/dooly/doolysim`,
`COPY` both from the archive, then the `ARG`, `RUN` and `LABEL` lines that follow them in
`docker/Dockerfile`, built with the same `--build-arg`.
