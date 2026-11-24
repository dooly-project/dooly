# Running DoolyProf with Docker

DoolyProf measures how long every GPU operation of an LLM takes when it is served by vLLM.
It works in two stages:

1. **Trace.** Run the model once in vLLM and record every distinct operation in its forward
   pass (matrix multiplies, attention, normalization, MoE routing, ...) together with how each
   operation's tensor shapes depend on the batch: the number of tokens, the number of requests,
   or the model's own sizes.
2. **Profile.** Time each operation on your GPU over a range of batch sizes, and store the
   results in a SQLite database. Optionally, also time the model's communication
   (all-reduce, all-gather, ...) on real groups of GPUs.

The Docker image has everything this needs: CUDA 13.0, Python 3.11, vLLM 0.33, PyTorch 2.14,
FlashInfer and DoolyProf. You only need a Linux machine with an NVIDIA GPU and this folder.
Model weights are never downloaded: DoolyProf only needs a model's config and tokenizer.

---

## What you need

| | |
|---|---|
| **GPU** | An NVIDIA GPU. Tested on A100 80GB and RTX PRO 6000 Blackwell. One GPU is enough; communication profiling needs one GPU per rank. |
| **Driver** | NVIDIA driver 580 or newer (CUDA 13.0). Check with `nvidia-smi`. |
| **Docker** | Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). |
| **Disk** | About 18 GB for the image (an 8 GB download). |
| **This repository** | For `docker/dooly-docker.sh`, the script that runs everything. |
| **python3** (optional) | Lets the script pick the best-connected GPUs. Without it, it takes the first free ones. |

If your account cannot run `docker` without `sudo`, run this once in your shell and everything
below works unchanged:

```bash
export DOCKER="sudo docker"
```

---

## Quick start

Run these from the root of this repository.

**1. Get the image** (an 8 GB download, once):

```bash
docker pull jhkjames1110/doolyprof
```

**2. Check that Docker can see your GPUs:**

```bash
docker run --rm --gpus all --entrypoint nvidia-smi jhkjames1110/doolyprof -L
```

You should see one line per GPU, e.g. `GPU 0: NVIDIA A100 80GB PCIe (UUID: ...)`. If you see
`could not select device driver`, the NVIDIA Container Toolkit is missing.

**3. Run the self-test** (about a minute, one GPU):

```bash
docker/dooly-docker.sh test
```

It ends with:

```
[dooly] 12/12 suites passed
```

**4. Download a model's config and tokenizer** (a few MB, no weights):

```bash
docker/dooly-docker.sh download Qwen/Qwen3-0.6B
```

**5. Trace and profile the model:**

```bash
docker/dooly-docker.sh run --model Qwen/Qwen3-0.6B --out /out/qwen3-0.6b
```

This takes about 3 minutes the first time. You will see (the two `[dooly] /opt/venv/...` lines
in between show the commands it runs):

```
[dooly-docker] GPUs 0 -> jhkjames1110/doolyprof:latest run --model Qwen/Qwen3-0.6B --out /out/qwen3-0.6b
[dooly] traced in 107s: /out/qwen3-0.6b/trace/Qwen_Qwen3-0.6B_1_dooly_trace.json
[dooly] profile exit 0 in 58s: 311 result rows, 0 ops without a working parent
```

`profile exit 0` and `0 ops without a working parent` mean every operation was timed.
The results are in `dooly-out/qwen3-0.6b/` on your machine (see [What you get](#what-you-get)).

The first run on a new GPU is slower: FlashInfer and Triton compile kernels and keep them in
`~/.cache/dooly-docker`, so later runs reuse them.

---

## Example workloads

`docker/example.sh` profiles three real models, one after another, each with a different
attention backend:

| Name | Model | What it shows | Attention backend |
|---|---|---|---|
| `dense` | `Qwen/Qwen3.8-27B`, tensor parallel 2 | A dense model (no experts) split in two; three of every four layers use linear attention (Gated DeltaNet), the fourth full attention. With two or more GPUs it also measures the all-reduces between the two ranks. | `FLASH_ATTN` |
| `moe` | `Qwen/Qwen3-30B-A3B` | A mixture of experts (128 experts, 8 used per token). | `FLASHINFER` |
| `hetero` | `openai/gpt-oss-20b` | Layers that alternate sliding-window and full attention. | `TRITON_ATTN` |

You need an 80 GB GPU (for Qwen3-30B-A3B), plus a second GPU for Qwen3.8-27B's communication.
None of the three models is gated, so no Hugging Face login is needed:

```bash
docker/example.sh                      # all three: ~15 minutes the first time, ~10 after
docker/example.sh moe hetero           # just these two
```

It ends with a summary like this one (RTX PRO 6000 Blackwell; parts left out as `...`). The
first run on a machine takes about 5 minutes longer while FlashInfer and Triton compile kernels.

```
=== summary (./dooly-out/example)
dense   Qwen/Qwen3.8-27B         460s   ok
moe     Qwen/Qwen3-30B-A3B       77s    ok
hetero  openai/gpt-oss-20b       66s    ok

dense: 10 operations to profile, 276 result rows; calls per forward pass:
    GemmaRMSNorm                                 129   2 kinds: 128 + 1
    aten::linear                                 129   3 kinds: 64 + 64 + 1
    _C::silu_and_mul                              64
    QwenGatedDeltaNetAttention                    48
    Qwen3NextAttention                            16
    ...
    all_reduce (group of 2)                      129   collective
    collective latencies measured: 2528 on nvidia-rtx-pro-6000-blackwell-server-edition-pcie, groups of 2
...
hetero: 13 operations to profile, 384 result rows; calls per forward pass:
    ...
    Attention                                     24   2 kinds: 12 + 12
    MoERunner                                     24
```

Operations with the same name but different shapes or settings are profiled separately and
shown as *kinds*. Qwen3.8-27B's two MLP linear layers per block are two kinds (plus the output
layer), and its attention is two operations: 48 linear-attention (Gated DeltaNet) layers and 16
full-attention layers. gpt-oss's attention comes in two kinds, sliding-window and full, 12 layers
each. Each model's results are in `dooly-out/example/<name>/`, laid out as described next;
running an example again replaces them.

---

## What you get

`/out/...` inside the container is `./dooly-out/...` on your machine. A run folder holds:

| File | What it is |
|---|---|
| `trace/<model>_<tp>_dooly_trace.json` | The trace: every operation the model runs and how its shapes depend on the batch. |
| `trace.log` | The tracer's log. Look here if the trace fails. |
| `profile.db` | **The results**, a SQLite database (tables below). |
| `profile.log` | The profiler's log. Look here if profiling fails. |
| `profile_timings.csv` | How long the profiler spent on each operation (for the profiler itself, not results). |

Main tables in `profile.db`:

| Table | One row per |
|---|---|
| `signatures` | Distinct operation (`operation_name`, plus its full description in `signature_json`). |
| `results` | Operation x batch configuration: `num_tokens`, `num_requests`, `is_prefill`, `kv_cache_size`, ... and the measured `mean/median/min/max/std_latency_ms`, with the GPU (`device`) and engine version (`engine`) it was measured on. |
| `profiled_models` | Model deployment: backend, tensor parallelism, data types, GPU, engine and backend version. |
| `model_operations` | How many times each operation runs in one forward pass of the model. |
| `coll_ops`, `coll_op_results` | The model's collectives, and (with `--profile-comm`) their latency by message size and group size. |
| `profile_runs` | Profiling run: GPU, engine, backend, package and driver versions, command line. |

For example, the median time of every operation at 64 tokens:

```bash
sqlite3 dooly-out/qwen3-0.6b/profile.db "
  SELECT s.operation_name, r.num_tokens, r.num_requests, r.is_prefill, r.median_latency_ms
  FROM results r JOIN signatures s USING (signature_hash)
  WHERE r.num_tokens = 64 ORDER BY r.median_latency_ms DESC"
```

A run reuses a result only from the same GPU and engine version, so runs on several machines
can share one database. To share results as files, `export` writes them as a tree of CSV files
and `import` merges one back into a database:

```
profiles/
├── compute/<gpu>/<engine>/<backend>/<model>/<deployment>/
│   ├── deployment.json   the deployment, and the runs that measured it (versions, options)
│   ├── model.csv         its operations: file, name, signature, calls per forward pass
│   ├── collectives.csv   its collectives: topology, group size, calls, how sizes scale
│   └── <operation>-<signature>.csv   latency of one operation per batch shape
└── communication/<topology>/<engine>/<collective>.csv   latency per group size and message size
```

---

## Commands

Every command goes through `docker/dooly-docker.sh`:

```
docker/dooly-docker.sh [--gpus N | --devices LIST] COMMAND [options]
```

| Command | What it does |
|---|---|
| `download MODEL [MODEL ...]` | Downloads config and tokenizer files from Hugging Face into your cache. Uses no GPU. |
| `run --model MODEL --out /out/NAME [options]` | Traces, then profiles. The usual command. |
| `test` | Runs DoolyProf's unit tests. |
| `check-update` | Says whether Docker Hub has a newer image than your machine. |
| `export --db /out/NAME/profile.db --out /out/profiles` | Writes the database's results as CSV files, one per operation, in a folder tree by GPU, engine, backend, model and deployment (below). Uses no GPU. |
| `import /out/profiles --db /out/NAME/profile.db` | Merges such a tree into a database, so later runs on the same GPU and engine reuse its results. Uses no GPU. |
| `verify /out/profiles` | Checks that a tree is exactly what `export` writes, e.g. before contributing it to [DoolyDB](https://huggingface.co/datasets/dooly-project/DoolyDB). Uses no GPU. |
| `trace ...` / `profile ...` | Runs only one stage, with the arguments of `doolyprof/run/run-tracer.py` / `run-profiler.py`. |

Options of `run`:

| Option | Default | Meaning |
|---|---|---|
| `--model` | (required) | Hugging Face model name, e.g. `meta-llama/Llama-3.1-8B`. Download it first. |
| `--out` | (required) | Output folder; start it with `/out/` so it lands in `./dooly-out/`. |
| `--db` | `OUT/profile.db` | Profile database to write to and reuse results from, e.g. one [DoolyDB](https://huggingface.co/datasets/dooly-project/DoolyDB) was imported into. |
| `--tp N` | 1 | Tensor parallelism: profile one rank of an N-way split. Runs on one GPU. |
| `--dp N` | 1 | Data parallelism (with `--enable-expert-parallel` for MoE expert parallelism). |
| `--enable-expert-parallel` | off | Split MoE experts across the `tp x dp` ranks. |
| `--attention-backend` | `FLASH_ATTN` | vLLM attention backend: `FLASH_ATTN`, `FLASHINFER`, `TRITON_ATTN`, or an MLA backend for MLA models. |
| `--max-batch-size` | 2 | Largest number of requests to profile. |
| `--max-seq-len` | 128 | Longest sequence to profile. |
| `--dtype` | `bfloat16` | Model data type. |
| `--gpu-memory-cap F` | | Fraction of GPU memory vLLM may use (raise it if the KV cache cannot hold the trace batch). |
| `--profile-comm` | off | Also time the model's collectives on real GPUs (see below). |
| `--tracer-args=...` / `--profiler-args=...` | | Extra arguments for one stage. Use the `=` form, e.g. `--profiler-args=--overwrite`. |

---

## Choosing GPUs

By default the script takes **one free GPU**. A GPU counts as free when it has less than 4 GB in
use, no other user's process, and no other run of this script on it (a run locks each GPU it takes
with `/tmp/dooly_gpu<i>.lock` until it ends). If none is free, it prints
`waiting for 1 free GPU(s)` and checks again every minute.

- `--gpus N` takes N free GPUs. It picks the N with the fastest links between them
  (NVLink before a shared PCIe switch, before crossing CPU sockets). If only worse-connected
  GPUs are free it waits for better ones; `DOOLY_ANY_LINK=1` takes them anyway.
- `--devices 2,3` uses exactly those GPUs (numbers as `nvidia-smi` shows them), in that order.
  It waits while one of them is busy.

Inside the container the chosen GPUs are numbered 0, 1, ... in the order chosen.

---

## Multi-GPU models and communication

`--tp`, `--dp` and `--enable-expert-parallel` describe how the model would be split. The
computation is always traced and profiled on **one** GPU, as one rank of that split, so you can
profile a 70B model at `--tp 4` on a single GPU.

To also time the communication between ranks, add `--profile-comm` and give the run as many
GPUs as ranks:

```bash
docker/dooly-docker.sh --gpus 2 run --model Qwen/Qwen3-0.6B --tp 2 --out /out/qwen3-0.6b-tp2 --profile-comm
```

The summary line then also says how many collective measurements were stored, e.g.
`collective measurements: 2898 on nvidia-rtx-pro-6000-blackwell-server-edition-pcie x2`.

- Measurements are labelled with the GPU model and whether the group is connected by NVLink,
  so results from different machines never get mixed up. Set `DOOLY_TOPOLOGY=mylabel` to choose
  the label yourself.
- Groups larger than the number of GPUs you gave are skipped with a warning.
- Communication is measured within one machine only.

---

## Models that need a Hugging Face login

Some models (Llama, Gemma, ...) require accepting a license on huggingface.co. Accept it on the
model's page, create a token at https://huggingface.co/settings/tokens, then:

```bash
HF_TOKEN=hf_xxx docker/dooly-docker.sh download meta-llama/Llama-3.1-8B
```

Only `download` uses the internet; tracing and profiling run offline from your cache.

---

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `DOCKER` | `docker` | The docker command, e.g. `sudo docker`. |
| `DOOLY_OUT` | `./dooly-out` | Host folder that appears as `/out` in the container. |
| `HF_CACHE` | `~/.cache/huggingface` | Host Hugging Face cache (models are downloaded here). |
| `DOOLY_CACHE` | `~/.cache/dooly-docker` | Host folder for compiled kernels. |
| `DOOLY_IMAGE` | `jhkjames1110/doolyprof:latest` | The image to run. |
| `HF_TOKEN` | | Hugging Face token, for `download` of gated models. |
| `DOOLY_ANY_LINK` | | `1`: take free GPUs even when better-connected ones exist but are busy. |
| `DOOLY_TOPOLOGY` | derived | Label for communication measurements. |
| `DOOLY_UPDATE_CHECK` | `1` | `0`: skip comparing your image with Docker Hub's before each command. |

---

## Troubleshooting

| You see | What to do |
|---|---|
| `permission denied ... docker.sock` | `export DOCKER="sudo docker"` |
| `could not select device driver "" with capabilities: [[gpu]]` | Install the NVIDIA Container Toolkit and restart Docker. |
| `MODEL is not in ... (the image works offline)` | Run `docker/dooly-docker.sh download MODEL` first. |
| `waiting for N free GPU(s)` | All GPUs are busy. Wait, or name GPUs with `--devices`. A run killed with `kill -9` leaves its lock behind: when no run is on GPU *i*, its owner removes it with `rmdir /tmp/dooly_gpu<i>.lock`. |
| `trace failed (exit N); see .../trace.log` | Read the end of `dooly-out/NAME/trace.log`. If vLLM says the attention backend is not supported on your GPU, try another `--attention-backend`. |
| `profile exit 1` or `ops without a working parent` above 0 | Read `dooly-out/NAME/profile.log`; please report it with the log. |
| KV cache too small for the trace batch | Add `--gpu-memory-cap 0.95`. |
| `argument --profiler-args: expected one argument` | Use the `=` form: `--profiler-args=--overwrite`. |
| `0 result rows` when running again with the same `--out` | Operations already in that folder's `profile.db` are not measured again (DoolyProf reuses what it has). Use a new `--out`, or add `--profiler-args=--overwrite` to measure again. |
| `Docker Hub has a newer jhkjames1110/doolyprof:latest than this machine` | Your copy of the image is older than the repository's main branch. Run the `docker pull` command it prints. |

---

## Updating, and building the image yourself

`jhkjames1110/doolyprof:latest` is the code on the repository's main branch. Docker downloads it
only when your machine has no copy, so before each command the script compares your copy with
Docker Hub's and, if Docker Hub's is newer, prints a warning with the command to update (it still
runs your copy; `DOOLY_UPDATE_CHECK=0` turns the check off). `docker/dooly-docker.sh check-update`
runs the check by itself. To update:

```bash
docker pull jhkjames1110/doolyprof
```

To see which commit your image was built from:

```bash
docker inspect --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' jhkjames1110/doolyprof
```

To build the image from this repository instead:

```bash
docker build -f docker/Dockerfile -t doolyprof:local .
DOOLY_IMAGE=doolyprof:local docker/dooly-docker.sh test
```
