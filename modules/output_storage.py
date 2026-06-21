import html
import os
from collections import defaultdict

from modules import shared


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".avif", ".gif", ".tif", ".tiff", ".bmp"}
DEFAULT_LOG_MAX_SIZE_MB = 10
DEFAULT_LOG_BACKUP_COUNT = 5


def format_bytes(size):
    size = float(size or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            if unit == "B":
                return f"{int(size)} B"
            return f"{size:.1f} {unit}"
        size /= 1024


def _option_number(opts, name, default, minimum=0):
    try:
        value = getattr(opts, name, default)
        value = float(value)
    except (TypeError, ValueError):
        value = default

    return max(minimum, value)


def configured_log_max_size_bytes(opts=None):
    opts = opts or shared.opts
    max_size_mb = _option_number(opts, "save_log_max_size_mb", DEFAULT_LOG_MAX_SIZE_MB)
    return int(max_size_mb * 1024 * 1024)


def configured_log_backup_count(opts=None):
    opts = opts or shared.opts
    return int(_option_number(opts, "save_log_backup_count", DEFAULT_LOG_BACKUP_COUNT))


def _log_backup_path(logfile_path, index):
    return f"{logfile_path}.{index}"


def rotate_log_file_if_needed(logfile_path, max_size_bytes=None, backup_count=None):
    max_size_bytes = configured_log_max_size_bytes() if max_size_bytes is None else max(0, int(max_size_bytes))
    backup_count = configured_log_backup_count() if backup_count is None else max(0, int(backup_count))

    if max_size_bytes <= 0 or not os.path.exists(logfile_path):
        return {"rotated": False, "removed": 0, "reason": "disabled_or_missing"}

    try:
        if os.path.getsize(logfile_path) < max_size_bytes:
            return {"rotated": False, "removed": 0, "reason": "under_limit"}
    except OSError:
        return {"rotated": False, "removed": 0, "reason": "unreadable"}

    removed = 0
    if backup_count <= 0:
        os.remove(logfile_path)
        return {"rotated": True, "removed": 1, "reason": "no_backups"}

    oldest = _log_backup_path(logfile_path, backup_count)
    if os.path.exists(oldest):
        os.remove(oldest)
        removed += 1

    for index in range(backup_count - 1, 0, -1):
        src = _log_backup_path(logfile_path, index)
        if os.path.exists(src):
            os.replace(src, _log_backup_path(logfile_path, index + 1))

    os.replace(logfile_path, _log_backup_path(logfile_path, 1))
    return {"rotated": True, "removed": removed, "reason": "rotated"}


def configured_output_dirs(opts=None):
    opts = opts or shared.opts
    names = [
        "outdir_samples",
        "outdir_txt2img_samples",
        "outdir_img2img_samples",
        "outdir_extras_samples",
        "outdir_grids",
        "outdir_txt2img_grids",
        "outdir_img2img_grids",
        "outdir_save",
        "outdir_init_images",
    ]

    result = []
    seen = set()
    for name in names:
        path = getattr(opts, name, "") or ""
        if not path:
            continue

        path = os.path.abspath(path)
        normalized = os.path.normcase(path)
        if normalized in seen:
            continue

        seen.add(normalized)
        result.append(path)

    return result


def _classify_file(path):
    filename = os.path.basename(path).lower()
    extension = os.path.splitext(filename)[1]

    if filename == "log.csv" or filename.startswith("log.csv."):
        return "logs"
    if extension == ".txt":
        return "txt"
    if extension in IMAGE_EXTENSIONS:
        return "images"
    return "other"


def summarize_output_storage(paths):
    summary = {
        "total_files": 0,
        "total_bytes": 0,
        "images_files": 0,
        "images_bytes": 0,
        "txt_files": 0,
        "txt_bytes": 0,
        "logs_files": 0,
        "logs_bytes": 0,
        "other_files": 0,
        "other_bytes": 0,
        "missing_dirs": [],
        "scanned_dirs": [],
        "top_dirs": [],
    }
    dir_sizes = defaultdict(int)

    for root in paths:
        if not root or not os.path.isdir(root):
            summary["missing_dirs"].append(root)
            continue

        summary["scanned_dirs"].append(root)
        for current_dir, _, filenames in os.walk(root):
            for filename in filenames:
                path = os.path.join(current_dir, filename)
                try:
                    size = os.path.getsize(path)
                except OSError:
                    continue

                category = _classify_file(path)
                summary["total_files"] += 1
                summary["total_bytes"] += size
                summary[f"{category}_files"] += 1
                summary[f"{category}_bytes"] += size
                dir_sizes[current_dir] += size

    summary["top_dirs"] = sorted(dir_sizes.items(), key=lambda item: item[1], reverse=True)[:5]
    return summary


def output_storage_report():
    paths = configured_output_dirs()
    summary = summarize_output_storage(paths)

    lines = [
        f"Output storage: {format_bytes(summary['total_bytes'])} in {summary['total_files']} files.",
        f"Images: {format_bytes(summary['images_bytes'])} in {summary['images_files']} files.",
        f"Text sidecars: {format_bytes(summary['txt_bytes'])} in {summary['txt_files']} files.",
        f"Save logs: {format_bytes(summary['logs_bytes'])} in {summary['logs_files']} files.",
    ]

    if summary["top_dirs"]:
        lines.append("Largest folders:")
        lines.extend(f"{format_bytes(size)} - {path}" for path, size in summary["top_dirs"])

    if summary["missing_dirs"]:
        lines.append("Skipped missing folders:")
        lines.extend(path for path in summary["missing_dirs"] if path)

    return "<br>".join(html.escape(line) for line in lines)
