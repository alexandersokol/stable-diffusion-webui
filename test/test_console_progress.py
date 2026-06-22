from types import SimpleNamespace

from modules import console_progress


class FakeProgressBar:
    def __init__(self):
        self.desc = ""
        self.bar_format = ""
        self.close_calls = 0
        self.refresh_calls = 0
        self.descriptions = []

    def close(self):
        self.close_calls += 1

    def refresh(self):
        self.refresh_calls += 1

    def set_description_str(self, desc, refresh=True):
        self.desc = desc
        self.descriptions.append((desc, refresh))


def test_progress_min_interval_uses_non_negative_float():
    opts = SimpleNamespace(console_progress_min_interval=1.25)

    assert console_progress.progress_min_interval(opts) == 1.25


def test_progress_min_interval_falls_back_for_invalid_values():
    opts = SimpleNamespace(console_progress_min_interval="bad")

    assert console_progress.progress_min_interval(opts) == console_progress.DEFAULT_MIN_INTERVAL


def test_image_count_label_clamps_current_to_total():
    state = SimpleNamespace(console_current_image_count=300, console_total_image_count=256)

    assert console_progress.image_count_label(state) == "[256/256]"


def test_generation_progress_desc_prefixes_image_count():
    state = SimpleNamespace(console_current_image_count=4, console_total_image_count=256)

    assert console_progress.generation_progress_desc(state) == "[4/256]"


def test_generation_progress_desc_omits_prefix_without_total():
    state = SimpleNamespace(console_current_image_count=4, console_total_image_count=0)

    assert console_progress.generation_progress_desc(state) == ""


def test_stage_label_uses_current_generation_stage():
    state = SimpleNamespace(console_generation_stage="DONE")

    assert console_progress.stage_label(state) == "[DONE]"


def test_stage_label_omits_empty_stage():
    state = SimpleNamespace(console_generation_stage="")

    assert console_progress.stage_label(state) == ""


def test_generation_progress_bar_format_adds_stage_to_end():
    state = SimpleNamespace(console_generation_stage="SAMPLING")

    assert console_progress.generation_progress_bar_format(state) == "{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}] [SAMPLING]"


def test_sampler_progress_kwargs_uses_generation_format():
    opts = SimpleNamespace(console_progress_min_interval=0.25)
    state = SimpleNamespace(
        console_current_image_count=1,
        console_total_image_count=100,
        console_generation_stage="SAMPLING",
    )

    kwargs = console_progress.sampler_progress_kwargs(opts, state)

    assert kwargs == {
        "desc": "[1/100]",
        "bar_format": "{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}] [SAMPLING]",
        "mininterval": 0.25,
    }


def test_generation_tqdm_context_injects_format_and_restores_targets():
    calls = []

    def fake_tqdm(*args, **kwargs):
        calls.append((args, kwargs))
        return "progress"

    target = SimpleNamespace(tqdm=fake_tqdm)
    opts = SimpleNamespace(console_progress_min_interval=0.25)
    state = SimpleNamespace(
        console_current_image_count=1,
        console_total_image_count=100,
        console_generation_stage="SAMPLING",
    )

    with console_progress.generation_tqdm_context(opts, state, targets=[(target, "tqdm")]):
        assert target.tqdm(range(3), disable=False) == "progress"

    assert calls == [(
        (range(0, 3),),
        {
            "disable": False,
            "desc": "[1/100]",
            "bar_format": "{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}] [SAMPLING]",
            "mininterval": 0.25,
        },
    )]
    assert target.tqdm is fake_tqdm


def test_generation_tqdm_context_keeps_explicit_kwargs():
    calls = []

    def fake_tqdm(*args, **kwargs):
        calls.append(kwargs)
        return "progress"

    target = SimpleNamespace(tqdm=fake_tqdm)
    opts = SimpleNamespace(console_progress_min_interval=0.25)
    state = SimpleNamespace(
        console_current_image_count=1,
        console_total_image_count=100,
        console_generation_stage="SAMPLING",
    )

    with console_progress.generation_tqdm_context(opts, state, targets=[(target, "tqdm")]):
        target.tqdm(range(3), desc="custom", bar_format="custom format", mininterval=2.0)

    assert calls == [{"desc": "custom", "bar_format": "custom format", "mininterval": 2.0}]


def test_tracked_generation_tqdm_repaints_stage_after_sampler_close():
    opts = SimpleNamespace(console_progress_min_interval=0.25)
    state = SimpleNamespace(
        console_current_image_count=2,
        console_total_image_count=256,
        console_generation_stage="SAMPLING",
    )
    bar = FakeProgressBar()

    tracked_bar = console_progress.track_generation_tqdm(bar)
    tracked_bar.close()
    assert bar.close_calls == 0

    state.console_generation_stage = "VAE"
    console_progress.refresh_generation_tqdm_bars(opts, state)

    assert bar.descriptions == [("[2/256]", False)]
    assert bar.bar_format == "{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}] [VAE]"
    assert bar.refresh_calls == 1
    assert bar.close_calls == 0

    state.console_generation_stage = "DONE"
    console_progress.close_generation_tqdm_bars(opts, state)

    assert bar.descriptions[-1] == ("[2/256]", False)
    assert bar.bar_format == "{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}] [DONE]"
    assert bar.close_calls == 1


def test_tracked_generation_tqdm_closes_with_original_image_and_done_stage():
    opts = SimpleNamespace(console_progress_min_interval=0.25)
    state = SimpleNamespace(
        console_current_image_count=1,
        console_total_image_count=4,
        console_generation_stage="SAMPLING",
    )
    bar = FakeProgressBar()

    console_progress.track_generation_tqdm(bar, desc=console_progress.generation_progress_desc(state))
    bar.close()
    state.console_current_image_count = 2
    state.console_generation_stage = "PREP"

    console_progress.close_generation_tqdm_bars(opts, state, stage="DONE")

    assert bar.descriptions[-1] == ("[1/4]", False)
    assert bar.bar_format == "{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}] [DONE]"
    assert bar.close_calls == 1


def test_wrap_tqdm_factory_tracks_progress_bar_results():
    opts = SimpleNamespace(console_progress_min_interval=0.25)
    state = SimpleNamespace(
        console_current_image_count=2,
        console_total_image_count=256,
        console_generation_stage="SAMPLING",
    )
    bar = FakeProgressBar()

    wrapped = console_progress.wrap_tqdm_factory(lambda *args, **kwargs: bar, opts, state)
    result = wrapped(range(3))
    result.close()

    assert result is bar
    assert bar.close_calls == 0

    console_progress.close_generation_tqdm_bars(opts, state)
    assert bar.close_calls == 1
