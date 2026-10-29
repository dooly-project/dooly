"""Call vLLM APIs whose keyword arguments differ between supported releases.

DoolyProf drives a few vLLM internals directly (``set_forward_context``,
``BlockTable``) and their keyword contracts change between releases: by 0.32
``set_forward_context`` no longer takes ``virtual_engine`` (pipeline virtual
engines were removed) and ``BlockTable`` no longer takes ``kernel_block_size``
(the allocation block size is the kernel block size). Rather than branch on
version strings, filter the keyword arguments against the callee's own
signature.

Only arguments that the callee does not declare are dropped, and only when the
caller marks them ``optional``; a required argument missing from the signature
raises so a renamed parameter is never silently lost.
"""
import functools
import inspect


@functools.lru_cache(maxsize=None)
def _accepted_params(fn):
    sig = inspect.signature(fn)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return None  # accepts anything
    return frozenset(sig.parameters)


def call_with_supported_kwargs(fn, *args, optional=(), **kwargs):
    """Call ``fn(*args, **kwargs)`` after dropping the ``optional`` keyword
    arguments that ``fn``'s signature does not declare."""
    accepted = _accepted_params(fn)
    if accepted is not None:
        for name in list(kwargs):
            if name not in accepted:
                if name not in optional:
                    raise TypeError(
                        f"{getattr(fn, '__qualname__', fn)} no longer accepts '{name}'; "
                        f"its parameters are {sorted(accepted)}")
                del kwargs[name]
    return fn(*args, **kwargs)


def set_forward_context(*args, **kwargs):
    """``vllm.forward_context.set_forward_context`` tolerant of the
    ``virtual_engine`` argument removed after vLLM 0.17."""
    from vllm.forward_context import set_forward_context as _sfc
    return call_with_supported_kwargs(_sfc, *args, optional=("virtual_engine",), **kwargs)


def make_block_table(**kwargs):
    """``vllm.v1.worker.block_table.BlockTable`` tolerant of the
    ``kernel_block_size`` argument removed after vLLM 0.17."""
    from vllm.v1.worker.block_table import BlockTable
    return call_with_supported_kwargs(BlockTable, optional=("kernel_block_size",), **kwargs)


def fill_slot_mapping(bt, num_reqs, req_indices, positions, query_start_loc):
    """Commit ``bt``'s block table and compute its slot mapping for one batch.

    Up to vLLM 0.17 ``BlockTable.compute_slot_mapping(req_indices, positions)``
    works on host numpy arrays and ``commit_slot_mapping`` copies the result to
    the GPU. In later releases (0.32) a GPU kernel computes it from ``(num_reqs,
    query_start_loc, positions)`` device tensors and reads the committed device
    block table directly, so the table is committed first. The contract is
    picked from the method's own parameter names. ``positions`` is the numpy
    position array; ``query_start_loc`` the int32 device tensor.
    """
    params = inspect.signature(bt.compute_slot_mapping).parameters
    if "query_start_loc" in params:
        import torch
        bt.commit_block_table(num_reqs)
        pos = torch.as_tensor(positions, dtype=torch.int64, device=query_start_loc.device)
        bt.compute_slot_mapping(num_reqs, query_start_loc, pos)
    else:
        bt.compute_slot_mapping(req_indices, positions)
        bt.commit_block_table(num_reqs)
        bt.commit_slot_mapping(len(positions))

