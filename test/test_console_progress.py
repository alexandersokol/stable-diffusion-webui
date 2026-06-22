from types import SimpleNamespace

from modules import console_progress


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
