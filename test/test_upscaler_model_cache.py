from modules import upscaler_model_cache


class FakeTensor:
    def __init__(self, numel, element_size):
        self._numel = numel
        self._element_size = element_size

    def numel(self):
        return self._numel

    def element_size(self):
        return self._element_size


class FakeModel:
    def __init__(self):
        self.moves = []
        self._parameters = [FakeTensor(16, 2)]
        self._buffers = [FakeTensor(8, 1)]

    def parameters(self):
        return iter(self._parameters)

    def buffers(self):
        return iter(self._buffers)

    def to(self, *args, **kwargs):
        self.moves.append((args, kwargs))
        return self


class FakeCpuFallbackModel(FakeModel):
    def to(self, *args, **kwargs):
        self.moves.append((args, kwargs))
        if kwargs.get("device") == "meta":
            raise RuntimeError("meta unavailable")
        return self


class FakeDescriptor:
    def __init__(self, model):
        self.model = model


def test_estimate_cached_model_bytes_uses_descriptor_model():
    model = FakeModel()
    descriptor = FakeDescriptor(model)

    assert upscaler_model_cache.estimate_cached_model_bytes(descriptor) == 40


def test_unload_cached_model_moves_descriptor_model_to_meta():
    original_torch_gc = upscaler_model_cache.devices.torch_gc
    calls = []
    upscaler_model_cache.devices.torch_gc = lambda: calls.append("gc")

    try:
        model = FakeModel()
        upscaler_model_cache.unload_cached_model(FakeDescriptor(model))

        assert model.moves == [((), {"device": "meta"})]
        assert calls == ["gc"]
    finally:
        upscaler_model_cache.devices.torch_gc = original_torch_gc


def test_unload_cached_model_falls_back_to_cpu_when_meta_fails():
    original_torch_gc = upscaler_model_cache.devices.torch_gc
    calls = []
    upscaler_model_cache.devices.torch_gc = lambda: calls.append("gc")

    try:
        model = FakeCpuFallbackModel()
        upscaler_model_cache.unload_cached_model(model)

        assert model.moves == [((), {"device": "meta"}), ((upscaler_model_cache.devices.cpu,), {})]
        assert calls == ["gc"]
    finally:
        upscaler_model_cache.devices.torch_gc = original_torch_gc
