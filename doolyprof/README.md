# DoolyProf

`DoolyProf` is a configuration-agnostic and redundancy aware profiler for LLM-inference workloads.
For any deployment configuration (e.g., hardware, attention backend implementation, serving engine, model), `DoolyProf` automatically detects the set of operations or modules present in the inference forward pass, and profiles each operation/module to output a SQLite database.
By detecting duplicate operations or modules across profile runs, `DoolyProf` allows for efficient profiling across different configurations. 
Downstream consumers (e.g., discrete-event simulator, prediction-based schedulers) can use to predict LLM serving performance across the large configuration space with minimal profiling overhead.

DoolyProf is one part of [Dooly](../README.md); [DoolySim](../doolysim/README.md) simulates
serving from the database it writes. Install both with the repository's `requirements.txt` (see
[Installation](../README.md#installation)), and run the commands below from the repository root.

## Run with Docker

The quickest way to try DoolyProf. The public image `jhkjames1110/doolyprof` has DoolyProf,
vLLM 0.33, PyTorch 2.14 and CUDA 13.0, and `docker/dooly-docker.sh` runs it on free GPUs.
You need Linux, an NVIDIA GPU with driver 580 or newer, and Docker with the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
git clone https://github.com/dooly-project/dooly.git
cd dooly
docker pull jhkjames1110/doolyprof
export HF_TOKEN_FILE=~/.hf-token.txt     # a file holding your Hugging Face token (docker/README.md)
docker/dooly-docker.sh download Qwen/Qwen3-0.6B                          # config and tokenizer, no weights
docker/dooly-docker.sh run --model Qwen/Qwen3-0.6B --out /out/qwen3-0.6b  # results: ./dooly-out/qwen3-0.6b/profile.db
```

`docker/example.sh` profiles three larger models: the dense Qwen3.8-27B at TP 2 together with
its all-reduces (three of every four of its layers use linear attention), the Qwen3-30B-A3B
mixture of experts, and gpt-oss-20b, whose layers alternate sliding-window and full attention.
[docker/README.md](../docker/README.md) explains every command, option and output;
[docker/AGENTS.md](../docker/AGENTS.md) is the version for coding agents.

## Quick start

### One-shot: trace + profile

```bash
python -m doolyprof.run.run-e2e \
    --model meta-llama/Llama-3.1-8B \
    --attention-backend FLASHINFER \
    --max-batch-size 32 \
    --max-seq-len 1024 \
    --tp 1 \
    --db-path ./dooly_data.db
```

This produces `./dooly_data.db`, a SQLite database keyed by `(model, attention_backend, tp_size, dtype, kv_cache_dtype)` containing per-operation latency samples.
Pass `--kv-cache-dtype fp8` (any vLLM KV-cache dtype literal) to profile a deployment with a quantized KV cache: only the attention operations get new signatures, every other operation is shared with the bf16-cache profile of the same model.

### Two-stage workflow

If you want to inspect the trace before profiling, or re-profile against an
existing trace:

```bash
# 1. Trace only — captures tensor dimension semantics
python -m doolyprof.run.run-tracer \
    --model meta-llama/Llama-3.1-8B

# 2. Profile using the trace
python -m doolyprof.run.run-profiler \
    --model meta-llama/Llama-3.1-8B \
    --traces ./vllm_traces_tp/traces-FLASHINFER-bfloat16/tp_1/meta-llama_Llama-3_1-8B/*.json \
    --attention-backend FLASHINFER \
    --max-batch-size 32 \
    --max-seq-len 1024 \
    --tp 1 \
    --db-path ./dooly_data.db
```

### Tensor parallelism

You can profile a TP>1 configuration on a single GPU using fake-TP:

```bash
python -m doolyprof.run.run-tracer --model meta-llama/Llama-3.1-70B --tp 4

```

The tracer sets `VLLM_FAKE_TP=4` and patches vLLM's `parallel_state` and collective ops to no-ops, so the model executes on one GPU while reporting TP=4 communication shapes. 

## Sharing results: export and import

Every result in the database records the GPU it was measured on (`device`, e.g.
`nvidia-rtx-pro-6000-blackwell-server-edition`) and the serving engine and version (`engine`,
e.g. `vllm-0.33.0`), and `profile_runs` records each run with its package and driver versions.
A run reuses a result only when both match, so one database can hold several GPUs and engine
versions. `DOOLY_DEVICE` and `DOOLY_ENGINE` override the two labels.

`export` writes a database as a tree of CSV files, one per operation, to share or keep under
version control; `import` merges such a tree into a database, after which runs on the same GPU
and engine reuse its results. `dooly run` also keeps its `trace.log`, `profile.log` and trace in
the database with the run (gzipped; home directory, user and host names written generically),
and `export` writes them next to the run's measurements as `runs/<run_uid>/*.gz`, so a run can be
debugged later from the tree alone:

```bash
python -m doolyprof.cli export --db ./dooly_data.db --out ./profiles
python -m doolyprof.cli import ./profiles --db ./dooly_data.db
python -m doolyprof.cli verify ./profiles     # the tree is exactly what export writes
```

`dooly run --db DB` profiles into DB instead of `OUT/profile.db`, reusing what it already has.
It runs on a GPU nobody else is on (under 4 GiB in use, no kernels running, no other user's
process, not taken by another `dooly run`), and with `--profile-comm` on one per rank of the largest group, as well
linked as the free ones allow (`--gpus N` for another count); it stops when too few are free.
`CUDA_VISIBLE_DEVICES` picks the GPUs instead. Without Docker, FlashInfer (used by vLLM) compiles
some kernels the first time they are used, with the nvcc of `CUDA_HOME`: point it at a toolkit of
the CUDA version PyTorch is built for (13.0 with `requirements.txt`); `run` prints a note when the
nvcc it finds is of another version.

With Docker: `docker/dooly-docker.sh export --db /out/NAME/profile.db --out /out/profiles`.

```
profiles/
├── compute/<gpu>/<engine>/<backend>/<model>/<deployment>/
│   ├── deployment.json   the deployment, and the runs that measured it (versions, options)
│   ├── model.csv         its operations: file, name, signature, calls per forward pass
│   ├── collectives.csv   its collectives: topology, group size, calls, how sizes scale
│   └── <operation>-<signature>.csv   latency of one operation per batch shape
└── communication/<topology>/<engine>/<collective>.csv   latency per group size and message size
```

For example `compute/nvidia-rtx-pro-6000-blackwell-server-edition/vllm-0.33.0/FLASH_ATTN-fa2/Qwen/Qwen3.8-27B/tp2-bfloat16-kvauto/`.
The backend folder names the version of what implements it (`FLASH_ATTN-fa2`,
`FLASHINFER-0.7.2.post1`, `TRITON_ATTN-3.8.0`). Each model folder is self-contained, so an
operation two models share is in both. An operation is measured once per batch shape: a
deployment of another tensor-parallel degree that has the same operation (the same shapes and
kernels, e.g. RMSNorm over the full hidden size) reuses it, and `num_tensor_parallel_workers`
records the degree of the run that measured it. Exporting into an existing tree replaces the
folders of the deployments the database has (and its collectives' files), and leaves the others;
importing never overwrites a result the database already has. Results measured before GPUs and
engines were recorded are not exported.

## Contributing to DoolyDB

[DoolyDB](https://huggingface.co/datasets/dooly-project/DoolyDB) is the shared tree of DoolyProf results on Hugging Face, in the layout
above. Start from it, so your run only measures what nobody has measured on your GPU and
engine yet, then send what you added as a pull request (step by step, with Docker and what to do
when `verify` fails: [docs/how-to-contribute.md](../docs/how-to-contribute.md)):

```bash
hf download dooly-project/DoolyDB --repo-type dataset --local-dir DoolyDB     # the current tree
python -m doolyprof.cli import DoolyDB --db dooly.db                          # into a database
python -m doolyprof.cli run --model ORG/MODEL --out runs/model --db dooly.db --max-batch-size 32 --max-seq-len 8192
python -m doolyprof.cli export --db dooly.db --out DoolyDB                    # your results join the tree
python -m doolyprof.cli verify DoolyDB                                        # exactly what export writes?
hf upload dooly-project/DoolyDB DoolyDB . --repo-type dataset --create-pr --commit-message "ORG/MODEL on <gpu>"
```

`hf` comes with `huggingface_hub` (`pip install huggingface_hub`); `hf auth login` first. `hf
upload` prints the URL of the commit it made, not of the pull request. The pull request opens as
a draft, which cannot be merged: find it on the dataset's Community tab
(Files changed lists the folders you added) and click **Publish**. Maintainers check it by
verifying its revision (`hf download dooly-project/DoolyDB --repo-type dataset --revision
refs/pr/<N> --local-dir pr-<N>`, then `python -m doolyprof.cli verify pr-<N>`; no GPU needed).
Check each `deployment.json` before uploading: it records the run's command line, with paths
under your home directory written as `~/…`, but other paths as they were. Profile at a scale DoolySim can
use (e.g. `--max-batch-size 32 --max-seq-len 8192`), and add `--tp N --profile-comm` on N
linked GPUs to include collectives. With Docker, everything runs through
`docker/dooly-docker.sh` (`download-db`, `import`, `run --db`, `export`, `verify`, `upload`), with
paths under `/out/` and your Hugging Face token in `HF_TOKEN_FILE`.

## Layout

```
doolyprof/
├── cli.py      # `python -m doolyprof.cli` (run, test); the Docker image's entry point
├── tracer/     # taint propagation through PyTorch ops
├── profiler/   # layer profiling and operation capture
├── run/        # entry points: run-tracer, run-profiler, run-e2e; collective profilers
└── tests/      # test suites (`python -m doolyprof.cli test`)
```
