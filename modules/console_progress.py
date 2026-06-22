DEFAULT_MIN_INTERVAL = 0.5
DEFAULT_BAR_FORMAT = "{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}]"


def progress_min_interval(opts) -> float:
    try:
        value = float(getattr(opts, "console_progress_min_interval", DEFAULT_MIN_INTERVAL))
    except (TypeError, ValueError):
        return DEFAULT_MIN_INTERVAL

    return max(0.0, value)


def image_count_label(state) -> str:
    total = int(getattr(state, "console_total_image_count", 0) or 0)
    current = int(getattr(state, "console_current_image_count", 0) or 0)

    if total <= 0:
        return ""

    current = max(0, min(current, total))
    return f"[{current}/{total}]"


def stage_label(state) -> str:
    stage = str(getattr(state, "console_generation_stage", "") or "").strip().upper()
    return f"[{stage}]" if stage else ""


def generation_progress_desc(state) -> str:
    return image_count_label(state)


def generation_progress_bar_format(state) -> str:
    stage = stage_label(state)
    return f"{DEFAULT_BAR_FORMAT} {stage}" if stage else DEFAULT_BAR_FORMAT
