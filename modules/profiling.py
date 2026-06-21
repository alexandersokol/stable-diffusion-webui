from pathlib import Path
import time

import torch

from modules import shared, ui_gradio_extensions


def format_bytes(size):
    if size >= 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024 * 1024):.1f} GB"

    return f"{size / (1024 * 1024):.1f} MB"


def trace_path(filename=None):
    filename = filename or shared.opts.profiling_filename
    return Path(filename).expanduser().resolve()


def rotated_trace_path(path, timestamp=None):
    timestamp = timestamp or time.strftime("%Y%m%d-%H%M%S")
    suffix = path.suffix
    stem = path.stem if suffix else path.name
    base = path.with_name(f"{stem}-{timestamp}{suffix}")

    if not base.exists():
        return base

    for index in range(1, 1000):
        candidate = path.with_name(f"{stem}-{timestamp}-{index}{suffix}")
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Unable to find an unused rotated profile filename for {path}")


def profile_trace_files(filename=None):
    path = trace_path(filename)
    directory = path.parent
    suffix = path.suffix
    stem = path.stem if suffix else path.name

    if not directory.exists():
        return []

    candidates = []
    if path.exists() and path.is_file():
        candidates.append(path)

    pattern = f"{stem}-*{suffix}" if suffix else f"{stem}-*"
    for candidate in directory.glob(pattern):
        if candidate == path or not candidate.is_file():
            continue
        candidates.append(candidate)

    return sorted(set(candidates), key=lambda x: x.stat().st_mtime, reverse=True)


def trace_files_size(files):
    return sum(x.stat().st_size for x in files if x.exists())


def rotate_existing_trace(filename=None):
    path = trace_path(filename)
    if not path.exists() or not path.is_file():
        return None

    destination = rotated_trace_path(path)
    path.rename(destination)
    return destination


def configured_max_trace_files():
    try:
        return max(0, int(getattr(shared.opts, "profiling_max_trace_files", 5)))
    except Exception:
        return 5


def configured_max_total_size_bytes():
    try:
        max_size_mb = int(getattr(shared.opts, "profiling_max_total_size_mb", 2048))
    except Exception:
        max_size_mb = 2048

    return max(0, max_size_mb) * 1024 * 1024


def cleanup_profile_traces(filename=None, max_files=None, max_total_size_bytes=None):
    max_files = configured_max_trace_files() if max_files is None else max(0, int(max_files))
    max_total_size_bytes = configured_max_total_size_bytes() if max_total_size_bytes is None else max(0, int(max_total_size_bytes))

    files = profile_trace_files(filename)
    before = trace_files_size(files)
    evicted = 0

    def over_limits(items):
        return (max_files > 0 and len(items) > max_files) or (max_total_size_bytes > 0 and trace_files_size(items) > max_total_size_bytes)

    while files and over_limits(files):
        victim = files[-1]
        try:
            victim.unlink()
            evicted += 1
        except FileNotFoundError:
            pass
        files = profile_trace_files(filename)

    after = trace_files_size(files)
    return {"before": before, "after": after, "evicted": evicted, "files": len(files)}


def cleanup_profile_traces_report():
    try:
        result = cleanup_profile_traces()
    except Exception as e:
        return f"Profiling trace cleanup failed: {e}"

    reclaimed = result["before"] - result["after"]
    return f"Profiling trace cleanup removed {result['evicted']} files and reclaimed {format_bytes(reclaimed)}. Current profiling trace size: {format_bytes(result['after'])}."


def export_chrome_trace(profiler, filename=None):
    path = trace_path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    rotate_existing_trace(path)
    profiler.export_chrome_trace(str(path))
    cleanup_profile_traces(path)


class Profiler:
    def __init__(self):
        if not shared.opts.profiling_enable:
            self.profiler = None
            return

        activities = []
        if "CPU" in shared.opts.profiling_activities:
            activities.append(torch.profiler.ProfilerActivity.CPU)
        if "CUDA" in shared.opts.profiling_activities:
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        if not activities:
            self.profiler = None
            return

        self.profiler = torch.profiler.profile(
            activities=activities,
            record_shapes=shared.opts.profiling_record_shapes,
            profile_memory=shared.opts.profiling_profile_memory,
            with_stack=shared.opts.profiling_with_stack
        )

    def __enter__(self):
        if self.profiler:
            self.profiler.__enter__()

        return self

    def __exit__(self, exc_type, exc, exc_tb):
        if self.profiler:
            shared.state.textinfo = "Finishing profile..."

            self.profiler.__exit__(exc_type, exc, exc_tb)

            export_chrome_trace(self.profiler)


def webpath():
    return ui_gradio_extensions.webpath(shared.opts.profiling_filename)

