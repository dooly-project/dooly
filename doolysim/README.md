# DoolySim

`DoolySim` is a discrete-event simulator for vLLM serving. It predicts end-to-end serving
performance (latency, throughput, per-request TTFT and TPOT) by running vLLM's own scheduler on
the CPU, so its batching decisions are vLLM's, and predicting each step's latency with
regression models trained on a DoolyProf database, without running the model on a GPU.

DoolySim is one part of [Dooly](../README.md); it reads a database made by
[DoolyProf](../doolyprof/README.md). Install both with the repository's `requirements.txt` (see
[Installation](../README.md#installation)), and run the commands below from the repository root.

## Configuration

DoolySim reads two paths repeatedly; set them once as environment variables, or pass
`--db-path` and `--cache-dir` to each command (which override the variables):

```bash
export DOOLYSIM_DB_PATH=/path/to/dooly_data.db        # DoolyProf database
export DOOLYSIM_CACHE_DIR=/path/to/prediction_cache/  # trained models and predictions
```

DoolySim's regression models do not extrapolate beyond the batch sizes and sequence lengths in
the database, so profile at the scale you will simulate (for example `--max-batch-size 32
--max-seq-len 8192`).

A database can hold results of several GPUs and engine versions (see
[DoolyProf](../doolyprof/README.md#sharing-results-export-and-import)). The estimator and the
simulator use the GPU and engine given with `--device` and `--engine`, by default the only ones
the database has for the deployment; collectives use the topology given with `--topology`, by
default the one the deployment's collectives were recorded under.

## Quick start

### 1. Train models and generate the prediction cache

```bash
python -m doolysim.estimator \
    --model meta-llama/Llama-3.1-8B \
    --backend FLASHINFER \
    --tp-size 1 \
    --cache-dir ./cache/tp1/llama-3.1-8b
```

A database may hold several deployments of one model (different TP degree, model dtype or
KV-cache dtype). `--tp-size`, `--dtype` and `--kv-cache-dtype` select which one to train on;
they default to all rows, bf16 and `auto`, mirroring the profiler's defaults. For TP > 1, add
`--train-comm` and `--topology` with the interconnect label the database's collective timings
were measured on.

This trains random-forest models from the profiled data, generates predictions across common
input ranges, and writes both `.pkl` (for retraining) and `.csv` (for fast simulation loading)
files into `--cache-dir`.

### 2. Generate a workload

`examples/workload/openchat-1rps.jsonl` is a ready workload (100 requests at 1 request/s). To
generate one from the openchat ShareGPT-4 dataset:

```bash
python -m doolysim.tools.generate_openchat_sharegpt4_dataset \
    --output examples/workload/openchat-1rps.jsonl \
    --num-requests 100 \
    --arrival-rate 1.0 \
    --tokenizer meta-llama/Llama-3.1-8B \
    --max-input-len 2048 \
    --max-output-len 512
```

The script downloads the dataset from Hugging Face, tokenizes each conversation, samples Poisson
arrivals at `--arrival-rate` requests/s, and writes a JSONL file with `{request_id, prompt,
prompt_len, output_len, arrival_time}` per line. `doolysim.tools.generate_sharegpt_dataset` does
the same from a local ShareGPT JSON file (`--sharegpt-path`).

### 3. Run a simulation

```bash
python -m doolysim.cli \
    --model meta-llama/Llama-3.1-8B \
    --backend FLASHINFER \
    --load-dataset examples/workload/openchat-1rps.jsonl
```

The simulator prints per-request latency, throughput, and aggregate metrics.

Useful flags:
- `--tp`, `--dtype` / `--kv-cache-dtype`: the deployment (vLLM literals, default `auto`). They
  configure the emulated engine and select the matching profiled rows, so an fp8-KV-cache
  deployment uses the attention latencies profiled with an fp8 cache.
- `--synthetic`: generate requests with vLLM's RandomDataset (`--input-len`, `--output-len`,
  `--num-requests`, `--arrival-rate`) instead of loading a file.
- `--json-output PATH`: write the results in vLLM's benchmark format, for comparison with
  `doolysim.benchmark.measure_vllm_streaming`, which measures real vLLM on the same workload.
- `--show-per-request`: show the per-request latency breakdown.

## Layout

```
doolysim/
├── cli.py          # `python -m doolysim.cli`: run a simulation
├── estimator.py    # `python -m doolysim.estimator`: train the latency models
├── core/           # simulation loop and step-latency predictor
├── emulator/       # fake CUDA/vLLM pieces for running vLLM's scheduler on the CPU
├── benchmark/      # real-vLLM measurement and simulation runners
├── analysis/       # result summaries
└── tools/          # workload generation and an alternative predictor
```
