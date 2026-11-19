"""Call vLLM APIs whose keyword arguments differ between supported releases.

DoolyProf drives a few vLLM internals directly (``set_forward_context``) and
their keyword contracts change between releases: by 0.32
``set_forward_context`` no longer takes ``virtual_engine`` (pipeline virtual
engines were removed). Rather than branch on version strings, filter the
keyword arguments against the callee's own signature. The attention
metadata's block table and slot mapping are built here directly
(``PagedBlockTable``), from the paged KV-cache layout, not with a model
runner's classes.

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


def call_module_with_recorded_arguments(module, args, kwargs):
    """Call ``module`` with the tensors the trace recorded for it, passing
    None for every parameter of its ``forward`` that has no default and no
    recorded value.

    The tracer records a module call's tensor arguments only, so an argument
    the engine passed as None leaves no trace: the first decoder layer's
    ``residual`` is None. While those parameters defaulted to None, the call
    worked without them; vLLM 0.34 makes some of them required
    (``Qwen3_5DecoderLayer.forward(positions, hidden_states, residual)``),
    and the layer could not be called at all."""
    fn = getattr(module, "forward", module)
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return module(*args, **kwargs)
    kwargs = dict(kwargs)
    P = inspect.Parameter
    positional = [p for p in params if p.kind in (P.POSITIONAL_ONLY, P.POSITIONAL_OR_KEYWORD)]
    for p in positional[len(args):] + [p for p in params if p.kind is P.KEYWORD_ONLY]:
        if p.default is P.empty and p.name not in kwargs and p.kind is not P.POSITIONAL_ONLY:
            kwargs[p.name] = None
    return module(*args, **kwargs)


def set_forward_context(*args, **kwargs):
    """``vllm.forward_context.set_forward_context`` tolerant of the
    ``virtual_engine`` argument removed after vLLM 0.17."""
    from vllm.forward_context import set_forward_context as _sfc
    return call_with_supported_kwargs(_sfc, *args, optional=("virtual_engine",), **kwargs)


class PagedBlockTable:
    """The block table and slot mapping of one KV-cache group, for the
    attention metadata the layer profiler builds.

    Both follow from vLLM's paged KV-cache layout alone: row ``r`` of the
    block table lists request ``r``'s block ids (int32), and the token at
    position ``p`` of request ``r`` is stored in slot
    ``block_table[r, p // block_size] * block_size + p % block_size`` (int64).
    vLLM's own block-table classes belong to a model runner and change with
    it: the runner's ``BlockTable`` dropped ``kernel_block_size`` by 0.32 and
    moved its slot mapping to a GPU kernel with another contract, and 0.34
    removed it together with that runner (``vllm.v1.worker.block_table``),
    which stopped every metadata build there. The successor
    (``vllm.v1.worker.gpu.block_table.BlockTables``) is built around the
    runner's staged batch writes. Writing the two tensors here reads the same
    on every release. ``block_table.gpu`` and ``slot_mapping.gpu`` are the
    device tensors, as on vLLM's classes.
    """

    class _Buffer:
        def __init__(self, gpu):
            self.gpu = gpu

    def __init__(self, block_size, max_num_reqs, max_num_blocks_per_req, max_num_batched_tokens,
                 device, **_unused):
        import numpy as np
        import torch
        self.block_size = int(block_size)
        self._rows = np.zeros((int(max_num_reqs), int(max_num_blocks_per_req)), dtype=np.int32)
        self.block_table = self._Buffer(torch.zeros(self._rows.shape, dtype=torch.int32, device=device))
        self.slot_mapping = self._Buffer(
            torch.zeros(int(max_num_batched_tokens), dtype=torch.int64, device=device))

    def clear(self):
        self._rows[:] = 0
        self.block_table.gpu.zero_()

    def add_row(self, block_ids, row_idx):
        self._rows[row_idx, :len(block_ids)] = block_ids

    def commit(self, num_reqs, req_indices, positions):
        """Copy the first ``num_reqs`` rows to the device and compute the slot
        of every token (``req_indices[i]``, ``positions[i]``)."""
        import numpy as np
        import torch
        rows = self._rows[:num_reqs]
        self.block_table.gpu[:num_reqs].copy_(torch.from_numpy(rows))
        req = np.asarray(req_indices, dtype=np.int64)
        pos = np.asarray(positions, dtype=np.int64)
        slots = rows[req, pos // self.block_size].astype(np.int64) * self.block_size + pos % self.block_size
        self.slot_mapping.gpu[:len(slots)].copy_(torch.from_numpy(slots))


def make_block_table(**kwargs):
    """A ``PagedBlockTable`` for ``block_size`` / ``max_num_reqs`` /
    ``max_num_blocks_per_req`` / ``max_num_batched_tokens`` / ``device``
    (other keyword arguments, which vLLM's classes took, are ignored)."""
    return PagedBlockTable(**kwargs)


def fill_slot_mapping(bt, num_reqs, req_indices, positions, query_start_loc=None):
    """Commit ``bt``'s first ``num_reqs`` rows and compute its slot mapping for
    the tokens at (``req_indices``, ``positions``), numpy arrays."""
    bt.commit(num_reqs, req_indices, positions)


class EngineSamplerCall:
    """Profiling callable for a model-runner sampler that is not an nn.Module.

    vLLM 0.32's default (V2) runner samples with ``sampler(logits,
    input_batch)``. The trace records only ``logits`` ([NUM_REQS, vocab]); the
    batch is runner state. This builds it the way the runner's own
    ``_dummy_sampler_run`` does, ``InputBatch.make_dummy(n, n,
    runner.input_buffers)`` with ``InputBatch`` taken from the runner's module,
    once per batch size (outside the timed call), and then calls
    ``sampler(logits, batch)``. Slots that never received a request sample
    greedily (temperature 0), the same path as the tracer's requests.
    """

    def __init__(self, sampler, model_runner):
        import sys
        self.sampler = sampler
        self.model_runner = model_runner
        self._input_batch_cls = getattr(sys.modules[type(model_runner).__module__], "InputBatch")
        self._batch = None
        self._num_reqs = None

    def __call__(self, logits):
        n = int(logits.shape[0])
        if n != self._num_reqs:
            self._batch = self._input_batch_cls.make_dummy(n, n, self.model_runner.input_buffers)
            self._num_reqs = n
        return self.sampler(logits, self._batch)

    def parameters(self):
        return iter(())
