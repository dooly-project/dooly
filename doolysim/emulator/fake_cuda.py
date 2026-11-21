import os
import torch


"""
Fake Tensor Parallelism (TP) for single-GPU profiling.

Patches vLLM's parallel_state, distributed communication, and linear layer
modules so that models behave as if running under TP > 1, while actually
executing on a single GPU.

IMPORTANT: This module must be imported BEFORE any vLLM modules so that
all cached references to parallel_state functions see the patched versions.

Usage:
    # In your entry-point script, set the env var before importing anything else:
    os.environ["VLLM_FAKE_TP"] = "2"

    # Then import this module to apply patches:
    from utns_profiler.profiler.utils.fake_tp import apply_fake_tp_patches, get_fake_tp_size
"""

_FAKE_CUDA_APPLIED = False

def get_fake_tp_size() -> int:
    """Return the fake TP size from the VLLM_FAKE_TP environment variable."""
    return int(os.environ.get("VLLM_FAKE_TP", "1"))


def apply_fake_tp_patches(tp_size: int) -> None:
    """
    Patch vLLM internals to simulate tensor parallelism on a single GPU.

    Patches three areas:
    1. parallel_state — world size, rank, and TP group
    2. distributed — all-reduce/all-gather become no-ops
    3. linear layers — world size and rank for column/row parallel

    Args:
        tp_size: The simulated tensor parallel world size (e.g. 2, 4, 8).
    """
    import vllm.distributed.parallel_state as ps
    from vllm import distributed
    from vllm.model_executor.layers import linear

    # --- Fake TP group ---------------------------------------------------
    class FakeTPGroup:
        world_size = tp_size
        rank_in_group = 0
        local_rank = 0
        rank = 0
        device = torch.device("cuda")
        cpu_group = None
        device_group = None

        def all_reduce(self, x):
            return ps.all_reduce_fake(
                tensor=x, group_name="fake_reduce"
            )

        def all_gather(self, x, dim=-1):
            return ps.all_gather_fake(
                tensor=x, dim=dim,
                world_size=self.world_size,
                group_name="fake_gather",
            )

        def reduce_scatter(self, tensor, dim, world_size, group_name):
            return ps.reduce_scatter_fake(
                tensor=tensor, dim=dim,
                world_size=world_size,
                group_name=group_name,
            )

        def patched_fused_scaled_matmul_reduce_scatter(self, x, y, z):
            return ps.patched_fused_scaled_matmul_reduce_scatter_fake(
                tensor=x, tensor2=y, tensor3=z,
                world_size=self.world_size,
                group_name="fake_fused_scaled_matmul_reduce_scatter",
            )

    fake_group = FakeTPGroup()

    # --- parallel_state --------------------------------------------------
    ps.get_tp_group = lambda: fake_group
    ps.get_tensor_model_parallel_world_size = lambda: tp_size
    ps.get_tensor_model_parallel_rank = lambda: 0

    # --- distributed communication (no-ops) ------------------------------
    distributed.tensor_model_parallel_all_reduce = lambda x: x
    distributed.tensor_model_parallel_all_gather = lambda x: x

    # --- linear layers ---------------------------------------------------
    linear.get_tensor_model_parallel_world_size = lambda: tp_size
    linear.get_tensor_model_parallel_rank = lambda: 0

    print(f"[FakeTP] Patches applied: TP={tp_size}")

class Dummy:
    def __init__(self, *args, **kwargs):
        # print("In init!")
        pass

    def __getattr__(self, name):
        def noop(*args, **kwargs):
            return None
        return noop
    
import torch.distributed.distributed_c10d as dc
    

class FakeBackendConfig(dc.BackendConfig):
    def get_device_backend_map(self) -> dict[str, dc.Backend]:
        backend_dict = super().get_device_backend_map()
        for key in backend_dict:
            if backend_dict[key] == dc.Backend.NCCL:
                backend_dict[key] = dc.Backend.GLOO
        return backend_dict

def apply_fake_backend_patches() -> None:
    dc.BackendConfig = FakeBackendConfig
    print(f'[Fake Backend] Patches applied!')



def _get_fake_gpu_total_memory() -> int:
    return int(os.environ.get("VLLM_FAKE_GPU_MEM_GB", "80")) * 1024**3

def _get_fake_gpu_free_memory() -> int:
    return int(os.environ.get("VLLM_FAKE_GPU_FREE_MEM_GB", "72")) * 1024**3

def _get_fake_gpu_device_count() -> int:
    return int(os.environ.get("VLLM_FAKE_GPU_DEVICE_COUNT", "1"))

def _get_fake_gpu_current_device() -> int:
    return int(os.environ.get("VLLM_FAKE_GPU_CURRENT_DEVICE", "0"))

def _get_fake_gpu_memory_reserved() -> int:
    return int(os.environ.get("VLLM_FAKE_GPU_MEM_RESERVED_GB", "0")) * 1024**3

def _get_fake_gpu_sm_count() -> int:
    return int(os.environ.get("VLLM_FAKE_GPU_SM_COUNT", "80"))

def _get_fake_gpu_cc() -> tuple[int, int]:
    major = int(os.environ.get("VLLM_FAKE_GPU_CC_MAJOR", "8"))
    minor = int(os.environ.get("VLLM_FAKE_GPU_CC_MINOR", "0"))
    return major, minor


class FakeProps:
    name = "FakeGPU"

    @property
    def major(self):
        return _get_fake_gpu_cc()[0]

    @property
    def minor(self):
        return _get_fake_gpu_cc()[1]

    @property
    def total_memory(self):
        return _get_fake_gpu_total_memory()

    @property
    def multi_processor_count(self):
        return _get_fake_gpu_sm_count()

    # Further fields of torch.cuda.get_device_properties() that newer vLLM reads
    is_integrated = False
    is_multi_gpu_board = False
    warp_size = 32
    L2_cache_size = 128 * 1024**2
    max_threads_per_multi_processor = 1536
    regs_per_multiprocessor = 65536
    shared_memory_per_block_optin = 99 * 1024
    shared_memory_per_multiprocessor = 100 * 1024
    uuid = "00000000-0000-0000-0000-000000000000"


def apply_fake_cuda():
    global _FAKE_CUDA_APPLIED
    if _FAKE_CUDA_APPLIED:
        return

    torch.cuda._lazy_init = lambda *a, **k: None
    torch.cuda.is_available = lambda: True
    torch.cuda.device_count = lambda: _get_fake_gpu_device_count()
    torch.cuda.current_device = lambda: _get_fake_gpu_current_device()
    torch.cuda.get_device_name = lambda i=0: "FakeGPU"
    torch.cuda.get_device_properties = lambda i=0: FakeProps()
    torch.cuda.empty_cache = lambda: None
    torch.cuda.mem_get_info = lambda *a, **k: (_get_fake_gpu_free_memory(), _get_fake_gpu_total_memory())
    torch.cuda.memory_reserved = lambda *a, **k: _get_fake_gpu_memory_reserved()
    torch.cuda.synchronize = lambda *a, **k: None

    orig_empty = torch.empty
    orig_ones = torch.ones
    orig_zeros = torch.zeros
    orig_zeros_like = torch.zeros_like
    orig_randn = torch.randn
    orig_tensor = torch.tensor
    orig_to = torch.Tensor.to
    orig_arange = torch.arange

    def is_cuda_device(value):
        if isinstance(value, torch.device):
            return value.type == "cuda"
        if isinstance(value, str):
            return value.startswith("cuda")
        # a bare index (e.g. current_platform.current_device()) means that CUDA device
        return isinstance(value, int) and not isinstance(value, bool)

    def redirect_device_arg(value):
        if is_cuda_device(value):
            return "cpu"
        return value

    def redirect_call(args, kwargs):
        redirected_args = list(args)
        redirected_kwargs = dict(kwargs)

        if redirected_args:
            redirected_args[0] = redirect_device_arg(redirected_args[0])

        if "device" in redirected_kwargs:
            redirected_kwargs["device"] = redirect_device_arg(
                redirected_kwargs["device"])

        return tuple(redirected_args), redirected_kwargs

    def redirect_device(kwargs):
        # device=None would be filled in by an active `with torch.device("cuda")` (vLLM builds
        # models under one), so it goes to the CPU too
        if "device" in kwargs and (kwargs["device"] is None or is_cuda_device(kwargs["device"])):
            kwargs["device"] = "cpu"
        return kwargs

    def fake_empty(*args, **kwargs):
        redirected_kwargs = redirect_device(kwargs)
        redirected_kwargs["pin_memory"] = False
        return orig_empty(*args, **redirected_kwargs)

    def fake_ones(*args, **kwargs):
        redirected_kwargs = redirect_device(kwargs)
        redirected_kwargs["pin_memory"] = False
        return orig_ones(*args, **redirected_kwargs)

    def fake_zeros(*args, **kwargs):
        redirected_kwargs = redirect_device(kwargs)
        redirected_kwargs["pin_memory"] = False
        return orig_zeros(*args, **redirected_kwargs)

    def fake_zeros_like(*args, **kwargs):
        redirected_kwargs = redirect_device(kwargs)
        redirected_kwargs["pin_memory"] = False
        return orig_zeros_like(*args, **redirected_kwargs)

    def fake_randn(*args, **kwargs):
        redirected_kwargs = redirect_device(kwargs)
        redirected_kwargs["pin_memory"] = False
        return orig_randn(*args, **redirected_kwargs)
    
    def fake_arange(*args, **kwargs):
        redirected_kwargs = redirect_device(kwargs)
        redirected_kwargs["pin_memory"] = False
        return orig_arange(*args, **redirected_kwargs)
    
    def fake_tensor(data, *args, **kwargs):
        redirected_kwargs = dict(kwargs)
        redirected_kwargs["device"] = redirect_device_arg(
            redirected_kwargs.get("device"))
        redirected_kwargs["pin_memory"] = False
        return orig_tensor(data, *args, **redirected_kwargs)
    
    def fake_to(self, *args, **kwargs):
        redirected_args, redirected_kwargs = redirect_call(args, kwargs)
        return orig_to(self, *redirected_args, **redirected_kwargs)

    def fake_cuda(self, device=None, non_blocking=False,
                  memory_format=torch.preserve_format):
        redirected_device = redirect_device_arg(device)
        return orig_to(
            self,
            device=redirected_device,
            non_blocking=non_blocking,
            memory_format=memory_format,
        )

    torch.empty = fake_empty
    torch.ones = fake_ones
    torch.zeros = fake_zeros
    torch.zeros_like = fake_zeros_like
    torch.randn = fake_randn
    torch.Tensor.to = fake_to
    torch.Tensor.cuda = fake_cuda
    torch.tensor = fake_tensor
    torch.arange = fake_arange
    torch.Tensor.uniform_ = lambda self, *args, **kwargs: self
    # The other tensor constructors that take a device: CUDA becomes CPU, as above.
    def cpu_constructor(fn):
        def constructor(*args, **kwargs):
            if "device" in kwargs:
                kwargs["device"] = "cpu" if kwargs["device"] is None else redirect_device_arg(kwargs["device"])
            if kwargs.get("pin_memory"):
                kwargs["pin_memory"] = False
            return fn(*args, **kwargs)
        return constructor
    for name in ("empty_like", "ones_like", "full", "full_like", "rand", "rand_like", "randn_like",
                 "randint", "randint_like", "empty_strided", "linspace", "logspace", "eye",
                 "as_tensor", "scalar_tensor", "tril_indices", "triu_indices"):
        if hasattr(torch, name):
            setattr(torch, name, cpu_constructor(getattr(torch, name)))
    for name in ("new_empty", "new_zeros", "new_ones", "new_full", "new_tensor", "new_empty_strided"):
        if hasattr(torch.Tensor, name):
            setattr(torch.Tensor, name, cpu_constructor(getattr(torch.Tensor, name)))

    # Pinned host memory needs a CUDA device; on the CPU a tensor is as good as pinned.
    torch.Tensor.pin_memory = lambda self, *args, **kwargs: self

    torch.cuda.memory_stats = lambda *a, **k: {"allocated_bytes.all.peak": 0}

    torch.cuda.set_device = lambda *a, **k: None
    torch.cuda.Stream = Dummy
    torch.cuda.Event = Dummy
    torch.Event = Dummy
    torch.cuda.current_stream = lambda *a, **k: Dummy()
    torch.cuda.reset_peak_memory_stats = lambda *a, **k: None    

    # Newer PyTorch has a device-generic API, torch.accelerator, that newer vLLM uses in place of
    # torch.cuda; give it the same fake device. Only the functions this PyTorch has are replaced.
    import contextlib
    fake_accelerator = {
        "is_available": lambda *a, **k: True,
        "device_count": lambda *a, **k: _get_fake_gpu_device_count(),
        "current_device_index": lambda *a, **k: _get_fake_gpu_current_device(),
        "current_accelerator": lambda *a, **k: torch.device("cuda"),
        "set_device_index": lambda *a, **k: None,
        "device_index": lambda *a, **k: contextlib.nullcontext(),
        "synchronize": lambda *a, **k: None,
        "current_stream": lambda *a, **k: Dummy(),
        "set_stream": lambda *a, **k: None,
        "empty_cache": lambda *a, **k: None,
        "empty_host_cache": lambda *a, **k: None,
        "get_memory_info": lambda *a, **k: (_get_fake_gpu_free_memory(), _get_fake_gpu_total_memory()),
        "memory_reserved": lambda *a, **k: _get_fake_gpu_memory_reserved(),
        "memory_allocated": lambda *a, **k: 0,
        "max_memory_allocated": lambda *a, **k: 0,
        "max_memory_reserved": lambda *a, **k: _get_fake_gpu_memory_reserved(),
        "memory_stats": lambda *a, **k: {"allocated_bytes.all.peak": 0},
        "reset_peak_memory_stats": lambda *a, **k: None,
    }
    accelerator = getattr(torch, "accelerator", None)
    if accelerator is not None:
        for name, fake in fake_accelerator.items():
            if hasattr(accelerator, name):
                setattr(accelerator, name, fake)

    apply_fake_tp_patches(get_fake_tp_size())
    apply_fake_backend_patches()
    _FAKE_CUDA_APPLIED = True