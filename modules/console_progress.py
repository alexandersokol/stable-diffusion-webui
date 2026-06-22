from contextlib import contextmanager


DEFAULT_MIN_INTERVAL = 0.5
DEFAULT_BAR_FORMAT = "{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}]"
_generation_tqdm_bars = []


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


def stage_label(state, stage=None) -> str:
    stage = str(getattr(state, "console_generation_stage", "") if stage is None else stage).strip().upper()
    return f"[{stage}]" if stage else ""


def generation_progress_desc(state) -> str:
    return image_count_label(state)


def generation_progress_bar_format(state, stage=None) -> str:
    stage = stage_label(state, stage=stage)
    return f"{DEFAULT_BAR_FORMAT} {stage}" if stage else DEFAULT_BAR_FORMAT


def sampler_progress_kwargs(opts, state) -> dict:
    return {
        "desc": generation_progress_desc(state),
        "bar_format": generation_progress_bar_format(state),
        "mininterval": progress_min_interval(opts),
    }


def wrap_tqdm_factory(factory, opts, state):
    def wrapped(*args, **kwargs):
        progress_kwargs = sampler_progress_kwargs(opts, state)
        for key, value in progress_kwargs.items():
            kwargs.setdefault(key, value)

        return track_generation_tqdm(factory(*args, **kwargs), desc=kwargs.get("desc"))

    return wrapped


def track_generation_tqdm(bar, desc=None):
    close = getattr(bar, "close", None)
    refresh = getattr(bar, "refresh", None)

    if close is None or refresh is None or getattr(bar, "_generation_tqdm_tracked", False):
        return bar

    entry = {
        "bar": bar,
        "close": close,
        "desc": desc,
        "close_requested": False,
        "closed": False,
    }

    def deferred_close():
        entry["close_requested"] = True

    bar.close = deferred_close
    bar._generation_tqdm_tracked = True
    bar._generation_tqdm_entry = entry
    _generation_tqdm_bars.append(entry)

    return bar


def _update_generation_tqdm_bar(entry, opts, state, stage=None):
    bar = entry["bar"]
    desc = entry["desc"] if entry["desc"] is not None else generation_progress_desc(state)
    bar_format = generation_progress_bar_format(state, stage=stage)

    if hasattr(bar, "set_description_str"):
        bar.set_description_str(desc, refresh=False)
    elif hasattr(bar, "desc"):
        bar.desc = desc

    if hasattr(bar, "bar_format"):
        bar.bar_format = bar_format

    bar.refresh()


def refresh_generation_tqdm_bars(opts, state):
    for entry in list(_generation_tqdm_bars):
        if entry["closed"]:
            continue

        _update_generation_tqdm_bar(entry, opts, state)


def close_generation_tqdm_bars(opts, state, stage=None):
    for entry in list(_generation_tqdm_bars):
        if entry["closed"]:
            continue

        _update_generation_tqdm_bar(entry, opts, state, stage=stage)

        bar = entry["bar"]
        bar.close = entry["close"]
        entry["closed"] = True
        entry["close"]()

    _generation_tqdm_bars[:] = [entry for entry in _generation_tqdm_bars if not entry["closed"]]


def default_generation_tqdm_targets():
    import tqdm
    import k_diffusion.sampling
    from modules import sd_samplers_lcm

    return [
        (tqdm, "tqdm"),
        (tqdm, "trange"),
        (k_diffusion.sampling, "tqdm"),
        (k_diffusion.sampling, "trange"),
        (sd_samplers_lcm, "trange"),
    ]


@contextmanager
def generation_tqdm_context(opts, state, targets=None):
    originals = []

    if targets is None:
        targets = default_generation_tqdm_targets()

    try:
        seen = set()
        for target, attr in targets:
            key = (id(target), attr)
            if key in seen:
                continue

            seen.add(key)
            original = getattr(target, attr, None)
            if original is None:
                continue

            originals.append((target, attr, original))
            setattr(target, attr, wrap_tqdm_factory(original, opts, state))

        yield
    finally:
        for target, attr, original in reversed(originals):
            setattr(target, attr, original)
