"""What ``dooly run`` decides before and after it measures, with NVML and psutil stood in for.

- GPUs: CUDA_VISIBLE_DEVICES wins; else the free GPUs by docker/dooly-docker.sh's rule (under
  4 GiB in use, no process of another user, no lock dir), each claimed with a lock dir; too few
  free GPUs stop the run and say which are busy and why
- memory already in use when an engine starts is blamed on other processes only when NVML shows
  other processes: this run's own earlier stages are named as such
- the command line a run records has the home directory as ~; the DoolyProf commit it records
  comes from git, or from doolyprof/REVISION where the code is not a checkout (the Docker image)
- the summary counts the rows this run wrote, not the database's

Runs directly: ``python doolyprof/tests/cli_run_test.py``. CPU only, standard library only.
"""
import contextlib
import io
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

from doolyprof import cli  # noqa: E402
from doolyprof.profiler import db as profile_db  # noqa: E402
from doolyprof.profiler.utils import gpu_memory  # noqa: E402

GiB = 1 << 30
ME = os.getuid()
OTHER = ME + 1  # another user
PARENT = 424242  # a process that started this one (an earlier stage of the run)


def _nvml(gpus):
    """A pynvml stand-in: ``gpus`` is [(bytes in use, [(pid, bytes)] or None when NVML cannot list)]."""
    m = types.ModuleType("pynvml")
    m.nvmlInit = m.nvmlShutdown = lambda: None
    m.nvmlDeviceGetCount = lambda: len(gpus)
    m.nvmlDeviceGetHandleByIndex = lambda i: i
    m.nvmlDeviceGetMemoryInfo = lambda h: types.SimpleNamespace(total=80 * GiB, free=80 * GiB - gpus[h][0],
                                                                used=gpus[h][0])

    def procs(h):
        if gpus[h][1] is None:
            raise RuntimeError("NVML: not supported")
        return [types.SimpleNamespace(pid=pid, usedGpuMemory=b) for pid, b in gpus[h][1]]
    m.nvmlDeviceGetComputeRunningProcesses = procs
    return m


def _psutil(users):
    """A psutil stand-in: ``users`` maps the PIDs visible here to their user id; PARENT started this process."""
    m = types.ModuleType("psutil")

    class Process:
        def __init__(self, pid=None):
            self.pid = os.getpid() if pid is None else pid
            if self.pid != os.getpid() and self.pid not in users:
                raise LookupError(f"no process {self.pid}")

        def uids(self):
            return types.SimpleNamespace(real=users.get(self.pid, ME))

        def parents(self):
            return [types.SimpleNamespace(pid=PARENT)]

        def children(self, recursive=False):
            return []
    m.Process = Process
    return m


@contextlib.contextmanager
def _machine(gpus, users=None, visible=None, lock_dir=None):
    """NVML and psutil showing ``gpus`` and ``users``; CUDA_VISIBLE_DEVICES ``visible`` (None: unset);
    lock dirs under ``lock_dir``; no nvidia-smi on PATH."""
    saved_mods = {k: sys.modules.get(k) for k in ("pynvml", "psutil")}
    saved_env = {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "PATH")}
    saved_lock = gpu_memory.LOCK
    sys.modules["pynvml"], sys.modules["psutil"] = _nvml(gpus), _psutil({PARENT: ME, **(users or {})})
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    if visible is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible
    empty = tempfile.mkdtemp()
    os.environ["PATH"] = empty
    if lock_dir:
        gpu_memory.LOCK = os.path.join(lock_dir, "dooly_gpu{}.lock")
    try:
        yield
    finally:
        gpu_memory.LOCK = saved_lock
        os.rmdir(empty)
        for k, v in saved_mods.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_free_gpus_by_the_wrappers_rule():
    with tempfile.TemporaryDirectory() as d:
        gpus = [(0, []),                          # free
                (6 * GiB, [(1001, 6 * GiB)]),     # another user's job
                (5 * GiB, [(2002, 5 * GiB)]),     # mine, but 5 GiB in use
                (0, []),                          # locked by another dooly run
                (GiB, [(2003, GiB)]),             # mine and small: free
                (GiB, [(9999, GiB)]),             # not visible here (a host PID in a container): by memory, free
                (5 * GiB, [(9998, 5 * GiB)])]     # not visible, 5 GiB in use: busy
        with _machine(gpus, users={1001: OTHER, 2002: ME, 2003: ME}, lock_dir=d):
            os.mkdir(gpu_memory.LOCK.format(3))
            free, busy = gpu_memory.free_gpus()
        assert free == [0, 4, 5], free
        assert busy[6] == "5.0 GiB in use", busy
        assert busy[1] == "processes of other users (pid 1001)", busy
        assert busy[2] == "5.0 GiB in use", busy
        assert busy[3].startswith("taken by another dooly run"), busy


def test_choose_gpus_takes_free_ones_and_locks_them():
    with tempfile.TemporaryDirectory() as d:
        with _machine([(0, [])] * 4, visible="3,2", lock_dir=d):   # chosen already (docker/dooly-docker.sh, the user)
            assert cli._choose_gpus(1) == ("3,2", [], None)
            assert os.listdir(d) == []
        with _machine([(6 * GiB, [(1001, 6 * GiB)]), (0, []), (0, [])], users={1001: OTHER}, lock_dir=d):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                gpus, locks, why = cli._choose_gpus(1)
            assert (gpus, why) == ("1", None), (gpus, why)
            assert locks == [gpu_memory.LOCK.format(1)] and os.path.isdir(locks[0]), locks
            assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
            assert "GPU(s) 1 (in use elsewhere: 0)" in out.getvalue(), out.getvalue()
            # GPU 1 is now taken: a second run gets GPU 2, a third finds none
            with contextlib.redirect_stdout(io.StringIO()):
                assert cli._choose_gpus(1)[0] == "2"
            gpus, locks2, why = cli._choose_gpus(1)
            assert gpus is None and locks2 == [], gpus
            assert "1 free GPU(s) needed, 0 free" in why and "GPU 0: processes of other users (pid 1001)" in why, why
            assert "CUDA_VISIBLE_DEVICES" in why, why
            cli._release(locks + [gpu_memory.LOCK.format(2)])
        assert os.listdir(d) == []


def test_memory_of_this_run_is_not_blamed_on_other_processes():
    def fraction(gpus, users=None):
        with _machine(gpus, users=users, visible="0"), contextlib.redirect_stdout(io.StringIO()) as out:
            frac = gpu_memory.engine_memory_fraction(cap=0.9, who="layer profiler")
        return frac, out.getvalue()

    frac, said = fraction([(int(2.5 * GiB), [(PARENT, int(2.5 * GiB))])])   # the earlier stage's tensors
    assert frac == round(77.5 / 80 - 0.1, 3), frac
    assert "by this run's earlier stages (cuda:0 2.5 GiB)" in said and "not reliable" not in said, said
    frac, said = fraction([(6 * GiB, [(1001, 6 * GiB)])], users={1001: OTHER})
    assert "cuda:0 6.0 GiB, 6.0 GiB of it by other processes" in said and "not reliable" in said, said
    frac, said = fraction([(int(2.5 * GiB), [])])   # a container: NVML lists no process, cannot tell whose
    assert "memory already in use (cuda:0 2.5 GiB)" in said and "not reliable" in said, said
    frac, said = fraction([(int(2.5 * GiB), [(9999, int(2.5 * GiB))])])   # a PID not visible here (host PID)
    assert "of it by other processes" not in said and "not reliable" in said, said
    frac, said = fraction([(0, [])])
    assert frac == 0.9 and said == "", (frac, said)
    try:
        fraction([(79 * GiB, [(1001, 79 * GiB)])], users={1001: OTHER})
        raise AssertionError("an engine was given a GPU another job fills")
    except RuntimeError as e:
        assert "not enough free GPU memory for the layer profiler: in use cuda:0 79.0 GiB" in str(e), e


def test_command_line_has_the_home_directory_as_tilde():
    argv = ["/home/u/dooly/doolyprof/run/run-profiler.py", "--db-path", "/home/u/dooly.db", "--out=/home/u/runs/x",
            "--traces", "/home/u/a.json,/home/u/b.json", "/home/user2/x", "/home/u", "/data/home/u/y", "--tp", "1"]
    assert profile_db.portable_argv(argv, home="/home/u/") == [
        "~/dooly/doolyprof/run/run-profiler.py", "--db-path", "~/dooly.db", "--out=~/runs/x",
        "--traces", "~/a.json,~/b.json", "/home/user2/x", "~", "/data/home/u/y", "--tp", "1"]
    assert profile_db.portable_argv(["/x"], home="/") == ["/x"]


def test_revision_without_git_comes_from_the_revision_file():
    saved = os.environ.pop("DOOLY_REVISION", None)
    try:
        with tempfile.TemporaryDirectory() as d:     # a copy of the package, not a git checkout
            assert profile_db._revision(d) is None
            with open(os.path.join(d, "REVISION"), "w") as f:
                f.write("0123abc\n")
            assert profile_db._revision(d) == "0123abc"
            os.environ["DOOLY_REVISION"] = "fedcba9"
            assert profile_db._revision(d) == "fedcba9"
    finally:
        os.environ.pop("DOOLY_REVISION", None)
        if saved is not None:
            os.environ["DOOLY_REVISION"] = saved


def test_summary_counts_the_rows_of_this_run():
    import profiles_test
    with tempfile.TemporaryDirectory() as d:
        db = f"{d}/p.db"
        assert cli._last_run(db) == 0
        profiles_test._fill(db, "gpu-x")          # an earlier run: 6 result rows, 4 collective rows
        before = cli._last_run(db)
        assert before == 1, before
        profiles_test._fill(db, "gpu-y")          # this run
        assert cli._run_rows(db, before) == (6, [("gpu-y-pcie", 2, 4)])
        assert cli._run_rows(db, 0)[0] == 12


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
