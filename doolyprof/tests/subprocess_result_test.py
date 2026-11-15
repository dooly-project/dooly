"""get_child_result returns a subprocess's result, and raises -- instead of
waiting forever -- when the subprocess dies before putting one (run-profiler
hung for 90 minutes when its resolver crashed at vLLM startup).

Runs directly: ``python doolyprof/tests/subprocess_result_test.py``. CPU only.
"""
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from doolyprof.profiler.utils.procs import get_child_result  # noqa: E402


def _ok(q):
    q.put({"resolved": [1, 2, 3]})


def _crash(q):
    raise ValueError("Free memory on device cuda:0 (27.77/79.25 GiB) on startup is less than desired")


def _killed(q):
    os._exit(9)


def _run(target):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=target, args=(q,))
    p.start()
    t0 = time.time()
    try:
        return get_child_result(p, q, "test child", poll_s=0.5), time.time() - t0
    finally:
        p.join()


def test_result_returned():
    out, _ = _run(_ok)
    assert out == {"resolved": [1, 2, 3]}, out


def test_crash_raises_instead_of_hanging():
    for target, code in ((_crash, 1), (_killed, 9)):
        try:
            _run(target)
        except RuntimeError as e:
            assert f"exited with code {code}" in str(e), str(e)
        else:
            raise AssertionError(f"{target.__name__}: no error raised")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
