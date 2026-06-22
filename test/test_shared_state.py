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


def test_console_generation_stage_resets_with_console_progress():
    state = State()

    state.set_console_generation_stage("SAMPLING")
    assert state.console_generation_stage == "SAMPLING"

    state.reset_console_image_progress()

    assert state.console_generation_stage == ""
