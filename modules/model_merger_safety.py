import os
import shutil
import tempfile


RAM_SAFETY_FACTOR = 1.35
DISK_SAFETY_FACTOR = 1.05


try:
    import psutil
except Exception:
    psutil = None


def format_bytes(size):
    size = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024


def existing_file_size(path):
    return os.path.getsize(path) if path and os.path.exists(path) else 0


def estimate_merge_memory_required(primary_path, secondary_path=None, tertiary_path=None, vae_path=None):
    primary_size = existing_file_size(primary_path)
    secondary_size = existing_file_size(secondary_path)
    tertiary_size = existing_file_size(tertiary_path)
    vae_size = existing_file_size(vae_path)

    merge_peak = primary_size + secondary_size
    diff_peak = secondary_size + tertiary_size
    vae_peak = primary_size + vae_size
    peak_size = max(primary_size, merge_peak, diff_peak, vae_peak)

    return int(peak_size * RAM_SAFETY_FACTOR)


def estimate_state_dict_size(state_dict):
    total = 0
    for value in state_dict.values():
        if hasattr(value, "numel") and hasattr(value, "element_size"):
            total += value.numel() * value.element_size()

    return total


def available_ram():
    if psutil is None:
        return None

    return psutil.virtual_memory().available


def check_available_ram(required_bytes, available_bytes=None):
    if available_bytes is None:
        available_bytes = available_ram()

    if available_bytes is not None and required_bytes > available_bytes:
        raise MemoryError(
            f"model merge needs about {format_bytes(required_bytes)} of available system RAM, "
            f"but only {format_bytes(available_bytes)} is available"
        )

    return available_bytes


def check_disk_space(output_path, required_bytes, free_bytes=None):
    output_dir = os.path.dirname(output_path) or "."
    required_bytes = int(required_bytes * DISK_SAFETY_FACTOR)

    if free_bytes is None:
        free_bytes = shutil.disk_usage(output_dir).free

    if required_bytes > free_bytes:
        raise OSError(
            f"model merge needs about {format_bytes(required_bytes)} of free disk space in {output_dir}, "
            f"but only {format_bytes(free_bytes)} is available"
        )

    return free_bytes


def atomic_write(output_path, writer):
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(prefix=f".{os.path.basename(output_path)}.", suffix=".tmp", dir=output_dir)
    os.close(fd)

    try:
        writer(temp_path)
        os.replace(temp_path, output_path)
    except Exception:
        try:
            os.remove(temp_path)
        except FileNotFoundError:
            pass
        raise
