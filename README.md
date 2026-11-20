# Dooly

Dooly predicts LLM serving performance across deployment configurations (hardware, attention
backend, model, parallelism) with little profiling. This repository has both of its parts:

- **DoolyProf** (`doolyprof/`) profiles a deployment into a SQLite database.
- **DoolySim** (`doolysim/`) simulates serving from that database.

`DoolyProf` is a configuration-agnostic and redundancy aware profiler for LLM-inference workloads.
For any deployment configuration (e.g., hardware, attention backend implementation, serving engine, model), `DoolyProf` automatically detects the set of operations or modules present in the inference forward pass, and profiles each operation/module to output a SQLite database.
By detecting duplicate operations or modules across profile runs, `DoolyProf` allows for efficient profiling across different configurations. 
Downstream consumers (e.g., discrete-event simulator, prediction-based schedulers) can use to predict LLM serving performance across the large configuration space with minimal profiling overhead.

`DoolySim` is a discrete-event simulator for vLLM serving. It predicts end-to-end serving
performance (latency, throughput, per-request TTFT and TPOT) by running vLLM's own scheduler on
the CPU, so its batching decisions are vLLM's, and predicting each step's latency with
regression models trained on a DoolyProf database, without running the model on a GPU.

Check our pre-print at [Dooly: Configuration-Agnostic, Redundancy-Aware Profiling for LLM Inference Simulation](https://arxiv.org/abs/2605.07985).

## Run with Docker

The quickest way to try DoolyProf. The public image `jhkjames1110/doolyprof` has DoolyProf,
vLLM 0.33, PyTorch 2.14 and CUDA 13.0, and `docker/dooly-docker.sh` runs it on free GPUs.
You need Linux, an NVIDIA GPU with driver 580 or newer, and Docker with the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
git clone https://github.com/dooly-project/dooly.git
cd dooly
docker pull jhkjames1110/doolyprof
docker/dooly-docker.sh download Qwen/Qwen3-0.6B                          # config and tokenizer, no weights
docker/dooly-docker.sh run --model Qwen/Qwen3-0.6B --out /out/qwen3-0.6b  # results: ./dooly-out/qwen3-0.6b/profile.db
```

`docker/example.sh` profiles three larger models: the dense Qwen3.8-27B at TP 2 together with
its all-reduces (three of every four of its layers use linear attention), the Qwen3-30B-A3B
mixture of experts, and gpt-oss-20b, whose layers alternate sliding-window and full attention.
[docker/README.md](docker/README.md) explains every command, option and output;
[docker/AGENTS.md](docker/AGENTS.md) is the version for coding agents.

## Installation

Requires Linux, Python 3.11 to 3.14, and an NVIDIA GPU with driver 580 or newer (CUDA 13.0).
DoolySim runs on the CPU and uses no GPU, but vLLM's CUDA build still needs the NVIDIA driver to
import. We test with vLLM 0.33.0, PyTorch 2.14.0 and Python 3.11, the versions
`requirements.txt` installs.

```bash
git clone https://github.com/dooly-project/dooly.git
cd dooly
pip install -r requirements.txt
```

## Profile with DoolyProf

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

## Simulate with DoolySim

DoolySim reads two paths repeatedly; set them once as environment variables, or pass
`--db-path` and `--cache-dir` to each command (which override the variables):

```bash
export DOOLYSIM_DB_PATH=/path/to/dooly_data.db        # DoolyProf database
export DOOLYSIM_CACHE_DIR=/path/to/prediction_cache/  # trained models and predictions
```

DoolySim's regression models do not extrapolate beyond the batch sizes and sequence lengths in
the database, so profile at the scale you will simulate (for example `--max-batch-size 32
--max-seq-len 8192`).

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

## Repository layout

```
dooly/
├── doolyprof/          # DoolyProf (import as `doolyprof.X`)
│   ├── tracer/         # taint propagation through PyTorch ops
│   ├── profiler/       # layer profiling and operation capture
│   └── run/            # entry points: run-tracer, run-profiler, run-e2e
├── doolysim/           # DoolySim (import as `doolysim.X`)
│   ├── core/           # simulation loop and step-latency predictor
│   ├── emulator/       # fake CUDA/vLLM pieces for running vLLM's scheduler on the CPU
│   ├── benchmark/      # real-vLLM measurement and simulation runners
│   ├── analysis/       # result summaries
│   ├── tools/          # workload generation and an alternative predictor
│   ├── cli.py          # `python -m doolysim.cli`
│   └── estimator.py    # `python -m doolysim.estimator`
├── docker/             # Docker image recipe, runner script (dooly-docker.sh), example.sh, docs
└── examples/
    ├── traces/         # example tainted traces (output of run-tracer, input of run-profiler)
    └── workload/       # example workload for DoolySim: openchat ShareGPT-4 at 1 request/s
```

## Citation
```
@misc{kim2026doolyconfigurationagnosticredundancyawareprofiling,
      title={Dooly: Configuration-Agnostic, Redundancy-Aware Profiling for LLM Inference Simulation}, 
      author={Joon Ha Kim and Geon-Woo Kim and Anoop Rachakonda and Daehyeok Kim},
      year={2026},
      eprint={2605.07985},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2605.07985}, 
}
```
## License
MIT — see [LICENSE](./LICENSE).
