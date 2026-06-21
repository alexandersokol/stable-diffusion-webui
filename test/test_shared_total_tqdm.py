from types import SimpleNamespace

from modules import shared_total_tqdm


class FakeTqdm:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.desc = kwargs.get("desc", "")
        self.total = kwargs.get("total")
        self.update_calls = 0
        self.descriptions = []
        self.closed = False
        FakeTqdm.instances.append(self)

    def update(self):
        self.update_calls += 1

    def set_description_str(self, desc, refresh=True):
        self.desc = desc
        self.descriptions.append((desc, refresh))

    def refresh(self):
        pass

    def close(self):
        self.closed = True


def setup_total_tqdm_state():
    shared_total_tqdm.shared.opts = SimpleNamespace(multiple_tqdm=True, console_progress_min_interval=1.25)
    shared_total_tqdm.shared.cmd_opts = SimpleNamespace(disable_console_progressbars=False)
    shared_total_tqdm.shared.progress_print_out = None
    shared_total_tqdm.shared.state = SimpleNamespace(
        job_count=8,
        sampling_steps=20,
        console_current_image_count=4,
        console_total_image_count=256,
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
        assert instance.kwargs["desc"] == "4/256 Total progress"
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

        assert FakeTqdm.instances[0].descriptions == [("8/256 Total progress", False)]
    finally:
        shared_total_tqdm.tqdm.tqdm = old_tqdm
