import sys
import types
from types import SimpleNamespace

import torch

sys.modules["modules.images"] = types.SimpleNamespace()

from modules import upscaler_utils


class RecordingModel:
    def __init__(self, offset=10):
        self.offset = offset
        self.input_devices = []

    def __call__(self, patch):
        self.input_devices.append(patch.device)
        return patch + self.offset


class RecordingTqdm:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.updates = 0
        RecordingTqdm.calls.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def update(self, value):
        self.updates += value


def setup_upscaler_state():
    upscaler_utils.shared.opts = SimpleNamespace(enable_upscale_progressbar=False)
    upscaler_utils.shared.state = SimpleNamespace(interrupted=False, skipped=False)


def test_tiled_upscale_2_accumulates_on_cpu_and_matches_expected_values():
    setup_upscaler_state()
    img = torch.arange(1 * 1 * 3 * 3, dtype=torch.float32).reshape(1, 1, 3, 3)
    model = RecordingModel(offset=10)

    output = upscaler_utils.tiled_upscale_2(
        img,
        model,
        tile_size=2,
        tile_overlap=1,
        scale=1,
        device=torch.device("cpu"),
        desc="test",
    )

    assert output.device == torch.device("cpu")
    assert torch.equal(output, img + 10)
    assert model.input_devices == [torch.device("cpu")] * 4


def test_tiled_upscale_2_uses_model_device_for_input_patches():
    setup_upscaler_state()
    img = torch.ones((1, 1, 2, 2), dtype=torch.float32)
    model = RecordingModel(offset=1)

    upscaler_utils.tiled_upscale_2(
        img,
        model,
        tile_size=1,
        tile_overlap=0,
        scale=1,
        device=torch.device("cpu"),
        desc="test",
    )

    assert model.input_devices == [torch.device("cpu")] * 4


def test_tiled_upscale_2_passes_console_progress_mininterval_to_tqdm():
    old_tqdm = upscaler_utils.tqdm.tqdm
    RecordingTqdm.calls = []
    upscaler_utils.tqdm.tqdm = RecordingTqdm
    try:
        upscaler_utils.shared.opts = SimpleNamespace(enable_upscale_progressbar=True, console_progress_min_interval=2.0)
        upscaler_utils.shared.state = SimpleNamespace(interrupted=False, skipped=False)

        img = torch.ones((1, 1, 2, 2), dtype=torch.float32)
        model = RecordingModel(offset=1)

        upscaler_utils.tiled_upscale_2(
            img,
            model,
            tile_size=1,
            tile_overlap=0,
            scale=1,
            device=torch.device("cpu"),
            desc="test",
        )

        assert RecordingTqdm.calls[0].kwargs["mininterval"] == 2.0
    finally:
        upscaler_utils.tqdm.tqdm = old_tqdm
