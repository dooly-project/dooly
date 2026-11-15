"""Waiting for a subprocess's result without hanging if it dies first."""
import queue as queue_mod


def get_child_result(proc, result_queue, what, poll_s=5.0):
    """``result_queue.get()`` while watching ``proc``: if it exits without having
    put a result (it crashed, or was killed), raise instead of waiting forever."""
    while True:
        try:
            return result_queue.get(timeout=poll_s)
        except queue_mod.Empty:
            if proc.is_alive():
                continue
            try:  # it may have put its result just before exiting
                return result_queue.get(timeout=poll_s)
            except queue_mod.Empty:
                raise RuntimeError(f"{what} (pid {proc.pid}) exited with code {proc.exitcode} before returning "
                                   f"its result; its error is printed above") from None
