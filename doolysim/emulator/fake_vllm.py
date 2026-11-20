from .fake_cuda import _get_fake_gpu_cc

def apply_vllm_patch():
    from vllm.platforms.cuda import DeviceCapability, NvmlCudaPlatform
    @classmethod
    def fake_get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
        return DeviceCapability(_get_fake_gpu_cc()[0], _get_fake_gpu_cc()[1])

    NvmlCudaPlatform.get_device_capability = fake_get_device_capability

    # Skip the memory-profiling forward pass of whichever model runner this vLLM uses (V1 and,
    # in newer vLLM, V2).
    import importlib
    for runner_module in ("vllm.v1.worker.gpu_model_runner", "vllm.v1.worker.gpu.model_runner"):
        try:
            importlib.import_module(runner_module).GPUModelRunner.profile_run = lambda *a, **k: None
        except (ImportError, AttributeError):
            pass

    # Newer vLLM reads small CPU buffers from the GPU through Unified Virtual Addressing views
    # (the V2 model runner does). With no GPU, the view of a CPU tensor is the tensor itself.
    # Modules that imported the helper by name before this point hold their own reference.
    try:
        from vllm.utils import torch_utils
        uva_view = getattr(torch_utils, "get_accelerator_view_from_cpu_tensor", None)
    except ImportError:
        uva_view = None
    if uva_view is not None:
        import sys
        cpu_view = lambda cpu_tensor, *a, **k: cpu_tensor
        for module in list(sys.modules.values()):
            if getattr(module, "get_accelerator_view_from_cpu_tensor", None) is uva_view:
                module.get_accelerator_view_from_cpu_tensor = cpu_view
    
    from vllm.v1.worker.gpu_worker import Worker
    # Nothing is compiled or warmed up. Newer vLLM returns the compile times as a named tuple.
    try:
        from vllm.v1.worker.worker_base import CompilationTimes
        no_compilation = CompilationTimes(**{f: 0.0 for f in CompilationTimes._fields})
    except ImportError:
        no_compilation = 0
    Worker.compile_or_warm_up_model = lambda *a, **k: no_compilation
    Worker.execute_dummy_batch = lambda *a, **k: None

    from vllm.platforms import current_platform
    current_platform.__class__.dist_backend = "gloo"

    # DoolySim does not run real kernels. Skip quantization weight-repack
    # paths that call CUDA-only kernels (e.g. gptq_marlin_repack) during
    # load. Required for gpt-oss-20b (MXFP4) under CUDA_VISIBLE_DEVICES="".
    try:
        from vllm.model_executor.layers.quantization.mxfp4 import (
            Mxfp4MoEMethod,
            XpuMxfp4MoEMethod,
        )
        Mxfp4MoEMethod.process_weights_after_loading = lambda *a, **k: None
        XpuMxfp4MoEMethod.process_weights_after_loading = lambda *a, **k: None
    except ImportError:
        pass

    print("[Fake vLLM] Pathces applied!")