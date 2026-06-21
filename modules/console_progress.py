DEFAULT_MIN_INTERVAL = 0.5


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
    return f"{current}/{total}"


def generation_progress_desc(state, base="Total progress") -> str:
    label = image_count_label(state)
    return f"{label} {base}" if label else base
