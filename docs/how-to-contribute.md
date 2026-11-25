# How to contribute to DoolyDB

[DoolyDB](https://huggingface.co/datasets/dooly-project/DoolyDB) is the shared set of DoolyProf
measurements on Hugging Face: the latency of every operation of a model's forward pass, by GPU,
serving engine and version, attention backend, model and deployment. Start from it, so that your
run only measures what nobody has measured on your GPU and engine yet, and send what you added
as a pull request.

DoolyDB is a folder of CSV and JSON files, in the layout `dooly export` writes
([doolyprof/README.md](../doolyprof/README.md#sharing-results-export-and-import)). The SQLite
database that DoolyProf and DoolySim work with is built from it on your machine with
`dooly import`, and turned back into files with `dooly export`.

## Before you start (once)

1. **Install Dooly**:

   ```bash
   git clone https://github.com/dooly-project/dooly.git
   cd dooly
   pip install -r requirements.txt
   ```

   You also need a CUDA toolkit (nvcc) of the CUDA version PyTorch is built for, 13.0 with
   `requirements.txt`: FlashInfer, which vLLM uses, compiles some kernels the first time they are
   used, with the nvcc of `CUDA_HOME`. Point it there, e.g.
   `export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH`; `dooly run`
   prints a note when the nvcc it finds is of another version.

   Or use the Docker image ([docker/README.md](../docker/README.md)), which has all of this; see
   [With Docker](#with-docker) for the same steps through `docker/dooly-docker.sh`.

2. **Install the Hugging Face command line and sign in**: `pip install huggingface_hub`, then
   `hf auth login`. Any Hugging Face account can open a pull request on DoolyDB; you do not need
   to be a member of `dooly-project`.

## Contribute a model

1. **Download the current DoolyDB**:

   ```bash
   hf download dooly-project/DoolyDB --repo-type dataset --local-dir DoolyDB
   ```

   `hf` keeps its download bookkeeping in `DoolyDB/.cache/huggingface/`; `verify` and `hf upload`
   ignore it.

2. **Import it into a local database**:

   ```bash
   python -m doolyprof.cli import DoolyDB --db dooly.db
   ```

   Importing only adds what the database does not have yet, so the `dooly.db` of an earlier
   contribution can be used again: download DoolyDB again and import it into the same file.

3. **Profile your model into that database**:

   ```bash
   python -m doolyprof.cli run --model ORG/MODEL --out runs/model --db dooly.db --max-batch-size 32 --max-seq-len 8192
   ```

   The run skips every operation `dooly.db` already has for your GPU and engine version and
   measures the rest; while DoolyDB has nothing for them yet, it measures everything. An
   operation is identified by its shapes and kernels, so a model shares most operations with its
   variants (on a GPU where Qwen2.5-7B was measured, Qwen2.5-7B-Instruct measures 1 of its 13)
   and few with other architectures. Profile at a scale DoolySim can use (e.g.
   `--max-batch-size 32 --max-seq-len 8192`), and add `--tp N --profile-comm` on N linked GPUs
   to include collectives.

   The run takes a GPU nobody else is on (under 4 GiB in use, no other user's process), or N
   linked ones with `--profile-comm`, and stops when there are too few; set
   `CUDA_VISIBLE_DEVICES` to choose them yourself. Attention runs on vLLM's `FLASH_ATTN` backend
   unless you pass `--attention-backend` (e.g. `FLASHINFER`, `TRITON_ATTN`); each backend gets
   its own folder in DoolyDB. The run ends with
   `[dooly] profile exit 0 in Ns: R result rows, 0 ops without a working parent`, where R is the
   number of rows it added to `dooly.db` (the profiler's own `Wrote N results` can be higher: a
   batch shape measured twice in one run is stored once).

4. **Export into the downloaded folder**:

   ```bash
   python -m doolyprof.cli export --db dooly.db --out DoolyDB
   ```

   The folders that were already there come out unchanged; your deployment gets its own folder,
   `compute/<gpu>/<engine>/<backend>/<org>/<model>/<deployment>/`. Each model folder is
   self-contained, so the files of operations it shares with other models are copied into it
   (copies of the existing measurements, not new ones).

5. **Check the folder**:

   ```bash
   python -m doolyprof.cli verify DoolyDB
   ```

   It ends with `[dooly] DoolyDB is valid: N deployment(s), ...` (the folder as you named it, e.g.
   `/out/DoolyDB` with Docker) when the folder is exactly what `dooly export` writes. See
   [If verify fails](#if-verify-fails) otherwise.

6. **Look over what you will upload**: the new folder should be your model's, and each new
   `deployment.json` records the run's command line. Paths under your home directory are
   written as `~/…`; other paths (e.g. under `/mnt`) are kept as they were. The folder also holds
   your run's logs and trace (`runs/<run_uid>/trace.log.gz`, `profile.log.gz`, `trace.json.gz`),
   so that the run can be debugged later; there your user and host names are written as `<user>`
   and `<host>` too. Read them with `zcat`.

7. **Upload it as a pull request**:

   ```bash
   hf upload dooly-project/DoolyDB DoolyDB . --repo-type dataset --create-pr --commit-message "ORG/MODEL on <gpu>"
   ```

   `hf` compares the folder with DoolyDB and uploads only the files that are new or changed. It
   prints the URL of the commit it made, not of the pull request; the pull request is on the
   dataset's Community tab, https://huggingface.co/datasets/dooly-project/DoolyDB/discussions.

8. **Publish the pull request**: it opens as a draft, which cannot be merged. On the dataset's
   page, open the **Community** tab and your pull request, check **Files changed** (your
   folders, nothing else), and click **Publish**. If a coding agent contributed for you, it
   leaves the pull request as a draft and reports its URL: this step is yours.

## With Docker

The same steps through `docker/dooly-docker.sh` (`./dooly-out/` on your machine is `/out/` in the
container), which needs no `hf` on your machine but always your token: a file that holds it in
`HF_TOKEN_FILE` (how to make one: [docker/README.md](../docker/README.md#what-you-need)), or the
token itself in `HF_TOKEN`. It uses that token, never a login saved on the machine. Use an image
from 2026-11-24 or later: `docker/dooly-docker.sh check-update` says whether yours is older, and
`docker pull jhkjames1110/doolyprof` updates it. On an account other people use too, point
`HF_CACHE` and `DOOLY_CACHE` at folders of your own first (the defaults are in that account's
home).

```bash
export HF_TOKEN_FILE=~/.hf-token.txt
docker/dooly-docker.sh download-db /out/DoolyDB
docker/dooly-docker.sh import /out/DoolyDB --db /out/dooly.db
docker/dooly-docker.sh download ORG/MODEL
docker/dooly-docker.sh run --model ORG/MODEL --out /out/model --db /out/dooly.db --max-batch-size 32 --max-seq-len 8192
docker/dooly-docker.sh export --db /out/dooly.db --out /out/DoolyDB
docker/dooly-docker.sh verify /out/DoolyDB
docker/dooly-docker.sh upload /out/DoolyDB "ORG/MODEL on <gpu>"
```

The image is offline, so `download` fetches the model's config and tokenizer first (no
weights). Add `--gpus N --tp N --profile-comm` to `run` to include collectives. `upload` runs
`verify` again, opens the pull request (never a commit to main) and prints its URL and what it
adds compared with DoolyDB. Then publish it as in step 8. A coding agent doing this follows
[docker/AGENTS.md](../docker/AGENTS.md#contributing-to-doolydb), which also has it check the
pull request itself.

## If verify fails

| `verify` says | Cause | Fix |
|---|---|---|
| `missing (model.csv lists it)` | a file of a deployment was deleted | export again (step 4) |
| `not written by dooly export` | a file `dooly export` does not write, e.g. notes | remove it |
| `differs from what dooly export writes` | a file was edited by hand, or two copies of an operation disagree | export again; if it persists, see below |
| `batch shape(s) measured more than once` | two measurements of one operation were put together by hand | import into a new database and export |

Two copies of an operation disagree when a pull request merged after your download measured an
operation you measured too (both of you added models that share it). Nothing needs measuring
again: put your folder on top of the current DoolyDB, so that the measurement already in DoolyDB
is kept and yours is dropped.

```bash
hf download dooly-project/DoolyDB --repo-type dataset --local-dir DoolyDB-new
python -m doolyprof.cli import DoolyDB-new --db new.db    # the current DoolyDB first
python -m doolyprof.cli import DoolyDB --db new.db        # then your folder: only what is new is added
python -m doolyprof.cli export --db new.db --out DoolyDB-new
python -m doolyprof.cli verify DoolyDB-new
```

Then upload `DoolyDB-new` as a new pull request (step 7) and close the old one.

## For maintainers

1. **Read Files changed**: a pull request should only add deployment folders (and rows to the
   collective files of its topology), not edit other files.
2. **Verify its revision** (no GPU needed):

   ```bash
   hf download dooly-project/DoolyDB --repo-type dataset --revision refs/pr/<N> --local-dir pr-<N>
   python -m doolyprof.cli verify pr-<N>
   ```

   The revision is DoolyDB as it would be after the merge, so this checks the result.
3. **Merge it.** The next contributor's download has it, and their runs skip its operations.
