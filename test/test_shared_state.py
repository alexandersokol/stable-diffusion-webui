import time

from modules import shared_state
from modules.shared_state import State


def test_end_clears_current_preview(monkeypatch):
    monkeypatch.setattr(shared_state.devices, "torch_gc", lambda: None)

    state = State()
    state.time_start = time.time()
    state.job = "task(txt2img-AAAAAAA)"
    state.job_count = 1
    state.current_latent = object()
    state.current_image = object()
    state.current_image_sampling_step = 12
    state.id_live_preview = 3

    state.end()

    assert state.current_latent is None
    assert state.current_image is None
    assert state.current_image_sampling_step == 0
    assert state.id_live_preview == 0
