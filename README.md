# Dooly

Dooly predicts LLM serving performance across deployment configurations (hardware, attention
backend, model, parallelism) with little profiling. This repository has both of its parts:

- **DoolyProf** ([`doolyprof/`](doolyprof/README.md)) profiles a deployment into a SQLite database.
- **DoolySim** ([`doolysim/`](doolysim/README.md)) simulates serving from that database.

Check our pre-print at [Dooly: Configuration-Agnostic, Redundancy-Aware Profiling for LLM Inference Simulation](https://arxiv.org/abs/2605.07985).

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

## Getting started

- **Profile a model** with DoolyProf: [doolyprof/README.md](doolyprof/README.md). The quickest
  way is the Docker image and `docker/dooly-docker.sh` ([docker/README.md](docker/README.md));
  the installed packages work too.
- **Simulate serving** with DoolySim: [doolysim/README.md](doolysim/README.md).

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
