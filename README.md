<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/dooly-project/dooly/main/assets/logos/dooly-logo-dark.png">
    <img alt="Dooly" src="https://raw.githubusercontent.com/dooly-project/dooly/main/assets/logos/dooly-logo-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Configuration-Agnostic, Redundancy-Aware Profiling for LLM Inference Simulation
</h3>

<p align="center">
| <a href="https://dooly-project.github.io"><b>Website</b></a> | <a href="https://arxiv.org/abs/2605.07985"><b>Paper</b></a> | <a href="https://huggingface.co/datasets/dooly-project/DoolyDB"><b>DoolyDB</b></a> |
</p>

Dooly provides a profiler that **automatically finds the set of operations** called by your deployment configuration (hardware, attention backend, model, parallelism),
and **reduces profiling time by re-using profiles** from the [DoolyDB](https://huggingface.co/datasets/dooly-project/DoolyDB).
As an example use case, we also provide a simulator (DoolySim) that predicts LLM serving performance across deployment configurations.

- **DoolyProf** ([`doolyprof/`](doolyprof/README.md)): our per-operation granularity profiler used to populate DoolyDB
- **DoolySim** ([`doolysim/`](doolysim/README.md)): emulates the source serving engine's scheduling logic and replaces GPU compute with latency predictions from models trained from DoolyDB

Check our paper at [Dooly: Configuration-Agnostic, Redundancy-Aware Profiling for LLM Inference Simulation](https://arxiv.org/abs/2605.07985) (NeuRIPS 2026).

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
- **Reuse and share profiles** with [DoolyDB](https://huggingface.co/datasets/dooly-project/DoolyDB):
  import it before profiling so that only what it does not have yet is measured, and send what
  you add as a pull request ([how](doolyprof/README.md#contributing-to-doolydb)).

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
