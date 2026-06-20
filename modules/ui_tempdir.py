import os
import json
import tempfile
import time
from collections import namedtuple
from pathlib import Path

import gradio.components

from PIL import PngImagePlugin

from modules import shared


Savedfile = namedtuple("Savedfile", ["name"])

WEBUI_TEMP_FILE_PREFIX = "sd-webui-"
WEBUI_TEMP_MANIFEST = ".sd-webui-temp-files.json"
WEBUI_TEMP_ORPHAN_TTL = 7 * 24 * 60 * 60


def _manifest_path(temp_dir):
    return os.path.join(temp_dir, WEBUI_TEMP_MANIFEST)


def _is_relative_to(path, parent):
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def _read_manifest(temp_dir):
    filename = _manifest_path(temp_dir)
    if not os.path.isfile(filename):
        return set()

    try:
        with open(filename, "r", encoding="utf8") as file:
            data = json.load(file)
    except Exception:
        return set()

    if isinstance(data, list):
        files = data
    else:
        files = data.get("files", [])

    return {os.path.abspath(file) for file in files if isinstance(file, str)}


def _write_manifest(temp_dir, filenames):
    os.makedirs(temp_dir, exist_ok=True)
    manifest = _manifest_path(temp_dir)
    fd, temp_manifest = tempfile.mkstemp(prefix=f".{WEBUI_TEMP_MANIFEST}.", suffix=".tmp", dir=temp_dir)

    try:
        with os.fdopen(fd, "w", encoding="utf8") as file:
            json.dump({"version": 1, "files": sorted(filenames)}, file)

        os.replace(temp_manifest, manifest)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.remove(temp_manifest)
        except FileNotFoundError:
            pass
        raise


def register_tmp_file_ownership(filename):
    temp_dir = shared.opts.temp_dir
    if temp_dir == "" or not _is_relative_to(filename, temp_dir):
        return

    filenames = _read_manifest(temp_dir)
    filenames.add(os.path.abspath(filename))
    _write_manifest(temp_dir, filenames)


def register_tmp_file(gradio, filename):
    if hasattr(gradio, 'temp_file_sets'):  # gradio 3.15
        gradio.temp_file_sets[0] = gradio.temp_file_sets[0] | {os.path.abspath(filename)}

    if hasattr(gradio, 'temp_dirs'):  # gradio 3.9
        gradio.temp_dirs = gradio.temp_dirs | {os.path.abspath(os.path.dirname(filename))}


def check_tmp_file(gradio, filename):
    if hasattr(gradio, 'temp_file_sets'):
        return any(filename in fileset for fileset in gradio.temp_file_sets)

    if hasattr(gradio, 'temp_dirs'):
        return any(Path(temp_dir).resolve() in Path(filename).resolve().parents for temp_dir in gradio.temp_dirs)

    return False


def save_pil_to_file(self, pil_image, dir=None, format="png"):
    already_saved_as = getattr(pil_image, 'already_saved_as', None)
    if already_saved_as and os.path.isfile(already_saved_as):
        register_tmp_file(shared.demo, already_saved_as)
        filename_with_mtime = f'{already_saved_as}?{os.path.getmtime(already_saved_as)}'
        register_tmp_file(shared.demo, filename_with_mtime)
        return filename_with_mtime

    if shared.opts.temp_dir != "":
        dir = shared.opts.temp_dir
    else:
        os.makedirs(dir, exist_ok=True)

    use_metadata = False
    metadata = PngImagePlugin.PngInfo()
    for key, value in pil_image.info.items():
        if isinstance(key, str) and isinstance(value, str):
            metadata.add_text(key, value)
            use_metadata = True

    file_obj = tempfile.NamedTemporaryFile(delete=False, prefix=WEBUI_TEMP_FILE_PREFIX, suffix=".png", dir=dir)
    try:
        pil_image.save(file_obj, pnginfo=(metadata if use_metadata else None))
    finally:
        file_obj.close()

    register_tmp_file_ownership(file_obj.name)
    register_tmp_file(shared.demo, file_obj.name)
    return file_obj.name


def install_ui_tempdir_override():
    """override save to file function so that it also writes PNG info"""
    gradio.components.IOComponent.pil_to_temp_file = save_pil_to_file


def on_tmpdir_changed():
    if shared.opts.temp_dir == "" or shared.demo is None:
        return

    os.makedirs(shared.opts.temp_dir, exist_ok=True)

    register_tmp_file(shared.demo, os.path.join(shared.opts.temp_dir, "x"))


def cleanup_tmpdr():
    temp_dir = shared.opts.temp_dir
    if temp_dir == "" or not os.path.isdir(temp_dir):
        return

    remaining_files = set()
    tracked_files = _read_manifest(temp_dir)

    for filename in tracked_files:
        if not _is_relative_to(filename, temp_dir):
            continue

        if not os.path.exists(filename):
            continue

        try:
            os.remove(filename)
        except OSError:
            remaining_files.add(filename)

    now = time.time()
    for root, _, files in os.walk(temp_dir, topdown=False):
        for name in files:
            if not name.startswith(WEBUI_TEMP_FILE_PREFIX):
                continue

            filename = os.path.join(root, name)
            if os.path.abspath(filename) in tracked_files:
                continue

            try:
                if now - os.path.getmtime(filename) < WEBUI_TEMP_ORPHAN_TTL:
                    continue

                os.remove(filename)
            except OSError:
                pass

    _write_manifest(temp_dir, remaining_files)


def is_gradio_temp_path(path):
    """
    Check if the path is a temp dir used by gradio
    """
    path = Path(path)
    if shared.opts.temp_dir and path.is_relative_to(shared.opts.temp_dir):
        return True
    if gradio_temp_dir := os.environ.get("GRADIO_TEMP_DIR"):
        if path.is_relative_to(gradio_temp_dir):
            return True
    if path.is_relative_to(Path(tempfile.gettempdir()) / "gradio"):
        return True
    return False
