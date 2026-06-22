from types import SimpleNamespace

import pytest

from modules import shared_total_tqdm


class FakeTqdm:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.desc = kwargs.get("desc", "")
        self.total = kwargs.get("total")
        self.bar_format = kwargs.get("bar_format")
        self.update_calls = 0
        self.descriptions = []
        self.refresh_calls = 0
        self.closed = False
        FakeTqdm.instances.append(self)

    def update(self):
        self.update_calls += 1

    def set_description_str(self, desc, refresh=True):
        self.desc = desc
        self.descriptions.append((desc, refresh))

    def refresh(self):
        self.refresh_calls += 1

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def restore_shared_state():
    original_state = shared_total_tqdm.shared.state
    original_opts = shared_total_tqdm.shared.opts
    original_cmd_opts = shared_total_tqdm.shared.cmd_opts
    original_progress_print_out = shared_total_tqdm.shared.progress_print_out
    try:
        yield
    finally:
        shared_total_tqdm.shared.state = original_state
        shared_total_tqdm.shared.opts = original_opts
        shared_total_tqdm.shared.cmd_opts = original_cmd_opts
        shared_total_tqdm.shared.progress_print_out = original_progress_print_out


def setup_total_tqdm_state():
    shared_total_tqdm.shared.opts = SimpleNamespace(multiple_tqdm=True, console_progress_min_interval=1.25)
    shared_total_tqdm.shared.cmd_opts = SimpleNamespace(disable_console_progressbars=False)
    shared_total_tqdm.shared.progress_print_out = None
    shared_total_tqdm.shared.state = SimpleNamespace(
        job_count=8,
        sampling_steps=20,
        console_current_image_count=4,
        console_total_image_count=256,
        console_generation_stage="SAMPLING",
    )


def test_total_tqdm_uses_image_count_description_and_mininterval():
    old_tqdm = shared_total_tqdm.tqdm.tqdm
    FakeTqdm.instances = []
    shared_total_tqdm.tqdm.tqdm = FakeTqdm
    try:
        setup_total_tqdm_state()
        total_tqdm = shared_total_tqdm.TotalTQDM()

        total_tqdm.update()

        instance = FakeTqdm.instances[0]
        assert instance.kwargs["desc"] == "[4/256]"
        assert instance.kwargs["bar_format"] == "{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}] [SAMPLING]"
        assert instance.kwargs["mininterval"] == 1.25
        assert instance.kwargs["total"] == 160
        assert instance.update_calls == 1
    finally:
        shared_total_tqdm.tqdm.tqdm = old_tqdm


def test_total_tqdm_refreshes_description_without_forcing_redraw():
    old_tqdm = shared_total_tqdm.tqdm.tqdm
    FakeTqdm.instances = []
    shared_total_tqdm.tqdm.tqdm = FakeTqdm
    try:
        setup_total_tqdm_state()
        total_tqdm = shared_total_tqdm.TotalTQDM()
        total_tqdm.update()

        shared_total_tqdm.shared.state.console_current_image_count = 8
        total_tqdm.update()

        assert FakeTqdm.instances[0].descriptions == [("[8/256]", False)]
    finally:
        shared_total_tqdm.tqdm.tqdm = old_tqdm


def test_total_tqdm_refreshes_stage_bar_format():
    old_tqdm = shared_total_tqdm.tqdm.tqdm
    FakeTqdm.instances = []
    shared_total_tqdm.tqdm.tqdm = FakeTqdm
    try:
        setup_total_tqdm_state()
        total_tqdm = shared_total_tqdm.TotalTQDM()
        total_tqdm.update()

        shared_total_tqdm.shared.state.console_generation_stage = "DONE"
        total_tqdm.update()

        instance = FakeTqdm.instances[0]
        assert instance.bar_format == "{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}] [DONE]"
        assert instance.refresh_calls == 1
    finally:
        shared_total_tqdm.tqdm.tqdm = old_tqdm
