from .fake_cuda import _get_fake_gpu_cc

def apply_vllm_patch():
    from vllm.platforms.cuda import DeviceCapability, NvmlCudaPlatform
    @classmethod
    def fake_get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
        return DeviceCapability(_get_fake_gpu_cc()[0], _get_fake_gpu_cc()[1])

    NvmlCudaPlatform.get_device_capability = fake_get_device_capability

    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    GPUModelRunner.profile_run = lambda *a, **k: None
    
    from vllm.v1.worker.gpu_worker import Worker
    Worker.compile_or_warm_up_model = lambda *a, **k: 0
    Worker.execute_dummy_batch = lambda *a, **k: None

    from vllm.platforms import current_platform                                                                                                                                                   
    current_platform.__class__.dist_backend = "gloo"

    print("[Fake vLLM] Pathces applied!")