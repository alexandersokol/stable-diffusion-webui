from __future__ import annotations

from pathlib import Path

from modules import ui_tempdir
from modules.paths_internal import script_path


def format_bytes(size):
    if size >= 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024 * 1024):.1f} GB"

    return f"{size / (1024 * 1024):.1f} MB"


def gradio_theme_cache_dir(root=None):
    root = Path(root or script_path)
    return root / "tmp" / "gradio_themes"


def moved_config_cache_files(root=None):
    root = Path(root or script_path)
    tmp_dir = root / "tmp"
    return [tmp_dir / "config.json", tmp_dir / "cache.json"]


def auxiliary_temp_files(root=None):
    files = []

    theme_dir = gradio_theme_cache_dir(root)
    if theme_dir.is_dir():
        files.extend(path for path in theme_dir.glob("*.json") if path.is_file())

    files.extend(path for path in moved_config_cache_files(root) if path.is_file())

    return sorted(set(files))


def cleanup_auxiliary_temp_files(root=None, include_ui_tempdir=True):
    files = auxiliary_temp_files(root)
    before = sum(path.stat().st_size for path in files if path.exists())
    removed = 0

    for path in files:
        try:
            path.unlink()
            removed += 1
        except FileNotFoundError:
            pass

    ui_tempdir_result = None
    if include_ui_tempdir:
        ui_tempdir_result = ui_tempdir.cleanup_tmpdir_report()

    return {
        "removed": removed,
        "reclaimed": before,
        "ui_tempdir": ui_tempdir_result,
    }


def cleanup_auxiliary_temp_files_report():
    try:
        result = cleanup_auxiliary_temp_files()
    except Exception as e:
        return f"Auxiliary temp cleanup failed: {e}"

    message = f"Auxiliary temp cleanup removed {result['removed']} files and reclaimed {format_bytes(result['reclaimed'])}."
    if result["ui_tempdir"]:
        message += f" {result['ui_tempdir']}"

    return message
