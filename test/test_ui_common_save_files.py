import importlib.util
import json
import sys
import types
import zipfile
from pathlib import Path


def load_ui_common_for_test(tmp_path, saved_files):
    import modules as modules_package

    ui_common_path = Path(__file__).resolve().parents[1] / "modules" / "ui_common.py"

    class FakeFile:
        @staticmethod
        def update(**kwargs):
            return kwargs

    class FakeFilenameGenerator:
        def __init__(self, *args, **kwargs):
            pass

        def apply(self, pattern):
            return "archive"

    def fake_save_image(*args, **kwargs):
        return str(saved_files[0]), str(saved_files[1])

    stubs = {
        "gradio": types.SimpleNamespace(File=FakeFile),
        "modules.call_queue": types.ModuleType("modules.call_queue"),
        "modules.output_storage": types.SimpleNamespace(rotate_log_file_if_needed=lambda *args, **kwargs: None),
        "modules.shared": types.SimpleNamespace(
            opts=types.SimpleNamespace(
                outdir_save=str(tmp_path),
                use_save_to_dirs_for_ui=False,
                samples_format="png",
                save_selected_only=False,
                save_write_log_csv=False,
                grid_zip_filename_pattern="",
            ),
            cmd_opts=types.SimpleNamespace(hide_ui_dir_config=True),
        ),
        "modules.ui_tempdir": types.ModuleType("modules.ui_tempdir"),
        "modules.util": types.ModuleType("modules.util"),
        "modules.infotext_utils": types.SimpleNamespace(
            image_from_url_text=lambda filedata: filedata,
            parse_generation_parameters=lambda text, skip_fields=None: {
                "Seed": "123",
                "Prompt": "prompt",
                "Negative prompt": "negative",
            },
        ),
        "modules.images": types.SimpleNamespace(
            save_image=fake_save_image,
            FilenameGenerator=FakeFilenameGenerator,
        ),
        "modules.ui_components": types.SimpleNamespace(ToolButton=object),
    }

    previous = {name: sys.modules.get(name) for name in stubs}
    previous_package_attrs = {}
    for name in stubs:
        if not name.startswith("modules."):
            continue

        attr_name = name.split(".", 1)[1]
        previous_package_attrs[attr_name] = (
            hasattr(modules_package, attr_name),
            getattr(modules_package, attr_name, None),
        )

    sys.modules.update(stubs)
    for name, module in stubs.items():
        if name.startswith("modules."):
            setattr(modules_package, name.split(".", 1)[1], module)

    try:
        spec = importlib.util.spec_from_file_location("_test_modules_ui_common", ui_common_path)
        ui_common = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ui_common)
        ui_common.modules = types.SimpleNamespace(images=stubs["modules.images"])
        return ui_common
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

        for attr_name, (had_attr, value) in previous_package_attrs.items():
            if had_attr:
                setattr(modules_package, attr_name, value)
            elif hasattr(modules_package, attr_name):
                delattr(modules_package, attr_name)


def make_generation_info():
    return json.dumps({
        "infotexts": ["prompt\nNegative prompt: negative\nSeed: 123"],
        "index_of_first_image": 0,
        "width": 16,
        "height": 16,
        "sampler_name": "Euler",
        "cfg_scale": 7,
        "steps": 20,
        "sd_model_name": "model",
        "sd_model_hash": "hash",
    })


def test_save_files_streams_zip_entries_with_zipfile_write(tmp_path, monkeypatch):
    saved_files = [tmp_path / "image.png", tmp_path / "image.txt"]
    saved_files[0].write_bytes(b"image-bytes")
    saved_files[1].write_text("parameters", encoding="utf8")
    ui_common = load_ui_common_for_test(tmp_path, saved_files)
    write_calls = []

    class RecordingZipFile:
        def __init__(self, filename, mode):
            self.filename = filename
            self.mode = mode

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def write(self, filename, arcname=None):
            write_calls.append((filename, arcname))

        def writestr(self, arcname, data):
            raise AssertionError("save_files should stream ZIP entries with ZipFile.write")

    monkeypatch.setattr(zipfile, "ZipFile", RecordingZipFile)

    file_update, html = ui_common.save_files(make_generation_info(), ["gallery-image"], True, -1)

    assert write_calls == [
        (str(saved_files[0]), "image.png"),
        (str(saved_files[1]), "image.txt"),
    ]
    assert file_update["visible"] is True
    assert file_update["value"][0] == str(tmp_path / "archive.zip")
    assert file_update["value"][1:] == [str(saved_files[0]), str(saved_files[1])]
    assert html == "<p>Saved: image.png</p>"
