from types import SimpleNamespace

from modules import devices


class FakeCuda:
    def __init__(self, *, free=900, total=1000, reserved=100, allocated=50):
        self.free = free
        self.total = total
        self.reserved = reserved
        self.allocated = allocated
        self.empty_cache_calls = 0
        self.ipc_collect_calls = 0

    def is_available(self):
        return True

    def mem_get_info(self):
        return self.free, self.total

    def memory_reserved(self):
        return self.reserved

    def memory_allocated(self):
        return self.allocated

    def empty_cache(self):
        self.empty_cache_calls += 1

    def ipc_collect(self):
        self.ipc_collect_calls += 1

    def device(self, _device):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def run_with_device_stubs(fake_cuda, *, now=100.0, opts=None, mps_gc=None):
    original_cuda = devices.torch.cuda
    original_monotonic = devices.time.monotonic
    original_opts = devices.shared.opts
    original_get_cuda_device_string = devices.get_cuda_device_string
    original_has_mps = devices.has_mps
    original_has_xpu = devices.has_xpu
    original_has_npu = devices.npu_specific.has_npu
    original_mac_specific = getattr(devices, "mac_specific", None)

    devices.torch_gc_last_run.clear()
    devices.torch.cuda = fake_cuda
    devices.time.monotonic = lambda: now
    devices.shared.opts = opts or SimpleNamespace(
        torch_gc_min_interval=2.0,
        torch_gc_cuda_free_memory_threshold=0.10,
        torch_gc_cuda_reserved_memory_threshold=0.25,
    )
    devices.get_cuda_device_string = lambda: "cuda"
    devices.has_mps = lambda: mps_gc is not None
    devices.has_xpu = lambda: False
    devices.npu_specific.has_npu = False
    devices.mac_specific = SimpleNamespace(torch_mps_gc=mps_gc or (lambda: None))

    def restore():
        devices.torch.cuda = original_cuda
        devices.time.monotonic = original_monotonic
        devices.shared.opts = original_opts
        devices.get_cuda_device_string = original_get_cuda_device_string
        devices.has_mps = original_has_mps
        devices.has_xpu = original_has_xpu
        devices.npu_specific.has_npu = original_has_npu
        if original_mac_specific is None:
            delattr(devices, "mac_specific")
        else:
            devices.mac_specific = original_mac_specific
        devices.torch_gc_last_run.clear()

    return restore


def test_torch_gc_skips_recent_cuda_cleanup_when_memory_is_healthy():
    fake_cuda = FakeCuda(free=900, total=1000, reserved=100, allocated=50)
    restore = run_with_device_stubs(fake_cuda, now=101.0)

    try:
        devices.torch_gc_last_run["cuda"] = 100.0
        devices.torch_gc()
    finally:
        restore()

    assert fake_cuda.empty_cache_calls == 0
    assert fake_cuda.ipc_collect_calls == 0


def test_torch_gc_force_runs_even_when_recent():
    fake_cuda = FakeCuda(free=900, total=1000, reserved=100, allocated=50)
    restore = run_with_device_stubs(fake_cuda, now=101.0)

    try:
        devices.torch_gc_last_run["cuda"] = 100.0
        devices.torch_gc(force=True)
    finally:
        restore()

    assert fake_cuda.empty_cache_calls == 1
    assert fake_cuda.ipc_collect_calls == 1


def test_torch_gc_runs_when_cuda_free_memory_is_low():
    fake_cuda = FakeCuda(free=50, total=1000, reserved=100, allocated=50)
    restore = run_with_device_stubs(fake_cuda, now=101.0)

    try:
        devices.torch_gc_last_run["cuda"] = 100.0
        devices.torch_gc()
    finally:
        restore()

    assert fake_cuda.empty_cache_calls == 1
    assert fake_cuda.ipc_collect_calls == 0


def test_torch_gc_runs_when_cuda_cached_reserved_memory_is_high():
    fake_cuda = FakeCuda(free=900, total=1000, reserved=500, allocated=100)
    restore = run_with_device_stubs(fake_cuda, now=101.0)

    try:
        devices.torch_gc_last_run["cuda"] = 100.0
        devices.torch_gc()
    finally:
        restore()

    assert fake_cuda.empty_cache_calls == 1
    assert fake_cuda.ipc_collect_calls == 0


def test_torch_gc_throttles_non_cuda_backends():
    calls = []
    fake_cuda = SimpleNamespace(is_available=lambda: False)
    restore = run_with_device_stubs(fake_cuda, now=101.0, mps_gc=lambda: calls.append("mps"))

    try:
        devices.torch_gc_last_run["mps"] = 100.0
        devices.torch_gc()
        devices.torch_gc(force=True)
    finally:
        restore()

    assert calls == ["mps"]
