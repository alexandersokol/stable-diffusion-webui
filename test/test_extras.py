import base64
import io
from pathlib import Path
from types import SimpleNamespace
from threading import Lock

import pytest
import requests
from PIL import Image
from fastapi import HTTPException

from modules import initialize

initialize.imports()

from modules import postprocessing
from modules import background_removal
from modules.api import api as api_module
from modules.api import models as api_models


def test_simple_upscaling_performed(base_url, img2img_basic_image_base64):
    payload = {
        "resize_mode": 0,
        "show_extras_results": True,
        "gfpgan_visibility": 0,
        "codeformer_visibility": 0,
        "codeformer_weight": 0,
        "upscaling_resize": 2,
        "upscaling_resize_w": 128,
        "upscaling_resize_h": 128,
        "upscaling_crop": True,
        "upscaler_1": "Lanczos",
        "upscaler_2": "None",
        "extras_upscaler_2_visibility": 0,
        "image": img2img_basic_image_base64,
    }
    assert requests.post(f"{base_url}/sdapi/v1/extra-single-image", json=payload).status_code == 200


def test_png_info_performed(base_url, img2img_basic_image_base64):
    payload = {
        "image": img2img_basic_image_base64,
    }
    assert requests.post(f"{base_url}/sdapi/v1/extra-single-image", json=payload).status_code == 200


def test_interrogate_performed(base_url, img2img_basic_image_base64):
    payload = {
        "image": img2img_basic_image_base64,
        "model": "clip",
    }
    assert requests.post(f"{base_url}/sdapi/v1/extra-single-image", json=payload).status_code == 200


class ExtrasTestState:
    def __init__(self):
        self.interrupted = False
        self.skipped = False
        self.stopping_generation = False
        self.textinfo = None
        self.job_count = 0

    def begin(self, job=""):
        self.job = job

    def end(self):
        self.job = ""

    def nextjob(self):
        pass

    def assign_current_image(self, image):
        self.current_image = image


def setup_postprocessing_unit_test(monkeypatch, tmp_path, *, script_run, output_extension=None):
    opts = SimpleNamespace(
        enable_pnginfo=False,
        list_hidden_files=False,
        outdir_extras_samples=str(tmp_path / "outputs"),
        outdir_samples="",
        postprocessing_existing_caption_action="Ignore",
        samples_format="png",
        use_original_name_batch=True,
    )

    monkeypatch.setattr(postprocessing.devices, "torch_gc", lambda: None)
    monkeypatch.setattr(postprocessing.shared, "opts", opts)
    monkeypatch.setattr(postprocessing, "opts", opts)
    monkeypatch.setattr(postprocessing.shared, "state", ExtrasTestState())
    monkeypatch.setattr(postprocessing.images, "read_info_from_image", lambda image: ("", {}))
    monkeypatch.setattr(
        postprocessing.scripts,
        "scripts_postproc",
        SimpleNamespace(
            run=script_run,
            output_extension=output_extension or (lambda args, default_extension: default_extension),
        ),
    )


def install_test_image_saver(monkeypatch):
    def save_image(image, path, basename, extension, forced_filename=None, suffix="", **kwargs):
        filename = forced_filename or f"{basename}{suffix}"
        output_file = Path(path) / f"{filename}.{extension}"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_file)
        return str(output_file), None

    monkeypatch.setattr(postprocessing.images, "save_image", save_image)


def test_postprocessing_recursively_mirrors_directory_tree(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    sources = [
        input_dir / "root.png",
        input_dir / "people" / "portrait.png",
        input_dir / "animals" / "portrait.png",
    ]
    for source in sources:
        source.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (1, 1), "red").save(source)

    setup_postprocessing_unit_test(monkeypatch, tmp_path, script_run=lambda pp, args: None)
    install_test_image_saver(monkeypatch)

    postprocessing.run_postprocessing(
        2, None, [], str(input_dir), str(output_dir), False, save_format="png"
    )

    assert sorted(path.relative_to(output_dir) for path in output_dir.rglob("*.png")) == [
        Path("animals/portrait.png"),
        Path("people/portrait.png"),
        Path("root.png"),
    ]
    assert postprocessing.shared.state.job_count == 3


def test_postprocessing_mirrors_tree_under_default_output_root(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    source = input_dir / "nested" / "source.png"
    source.parent.mkdir(parents=True)
    Image.new("RGB", (1, 1), "red").save(source)

    setup_postprocessing_unit_test(monkeypatch, tmp_path, script_run=lambda pp, args: None)
    install_test_image_saver(monkeypatch)

    postprocessing.run_postprocessing(
        2, None, [], str(input_dir), "", False, save_format="png"
    )

    assert (tmp_path / "outputs" / "nested" / "source.png").is_file()


def test_postprocessing_excludes_nested_output_directory(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    output_dir = input_dir / "output"
    source = input_dir / "source.png"
    prior_output = output_dir / "prior.png"
    output_dir.mkdir(parents=True)
    Image.new("RGB", (1, 1), "red").save(source)
    Image.new("RGB", (1, 1), "blue").save(prior_output)

    setup_postprocessing_unit_test(monkeypatch, tmp_path, script_run=lambda pp, args: None)
    install_test_image_saver(monkeypatch)

    postprocessing.run_postprocessing(
        2, None, [], str(input_dir), str(output_dir), False, save_format="png"
    )

    assert postprocessing.shared.state.job_count == 1
    assert sorted(path.name for path in output_dir.glob("*.png")) == ["prior.png", "source.png"]


def test_postprocessing_excludes_nested_link_to_external_output(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    output_target = tmp_path / "external-output"
    output_link = input_dir / "output"
    source = input_dir / "source.png"
    prior_output = output_target / "prior.png"
    input_dir.mkdir()
    output_target.mkdir()
    Image.new("RGB", (1, 1), "red").save(source)
    Image.new("RGB", (1, 1), "blue").save(prior_output)

    try:
        output_link.symlink_to(output_target, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")

    setup_postprocessing_unit_test(monkeypatch, tmp_path, script_run=lambda pp, args: None)
    install_test_image_saver(monkeypatch)

    postprocessing.run_postprocessing(
        2, None, [], str(input_dir), str(output_link), False, save_format="png"
    )

    assert postprocessing.shared.state.job_count == 1
    assert (output_target / "source.png").is_file()
    assert not (output_target / "output" / "prior.png").exists()


def test_postprocessing_processes_nested_sources_when_input_equals_output(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    source = input_dir / "nested" / "source.png"
    source.parent.mkdir(parents=True)
    Image.new("RGB", (1, 1), "red").save(source)

    setup_postprocessing_unit_test(monkeypatch, tmp_path, script_run=lambda pp, args: None)
    install_test_image_saver(monkeypatch)

    postprocessing.run_postprocessing(
        2, None, [], str(input_dir), str(input_dir), False, save_format="png"
    )

    assert postprocessing.shared.state.job_count == 1
    assert source.is_file()


def test_postprocessing_skips_existing_output_in_mirrored_directory(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    source = input_dir / "nested" / "source.jpg"
    existing_output = output_dir / "nested" / "source.png"
    source.parent.mkdir(parents=True)
    existing_output.parent.mkdir(parents=True)
    Image.new("RGB", (1, 1), "red").save(source)
    Image.new("RGB", (1, 1), "blue").save(existing_output)

    script_calls = []
    setup_postprocessing_unit_test(
        monkeypatch,
        tmp_path,
        script_run=lambda pp, args: script_calls.append(pp),
        output_extension=lambda args, default_extension: "png",
    )
    install_test_image_saver(monkeypatch)

    postprocessing.run_postprocessing(
        2,
        None,
        [],
        str(input_dir),
        str(output_dir),
        False,
        save_format="jpg",
        skip_existing_files=True,
    )

    assert postprocessing.shared.state.job_count == 1
    assert script_calls == []


def test_postprocessing_skips_existing_batch_output_when_enabled(monkeypatch, tmp_path):
    input_file = tmp_path / "source.png"
    output_dir = tmp_path / "outputs"
    input_file.parent.mkdir(exist_ok=True)
    output_dir.mkdir()
    Image.new("RGB", (1, 1), "red").save(input_file)
    (output_dir / "source.jpg").write_bytes(b"already here")

    script_calls = []
    save_calls = []
    setup_postprocessing_unit_test(monkeypatch, tmp_path, script_run=lambda pp, args: script_calls.append(pp))
    monkeypatch.setattr(postprocessing.images, "save_image", lambda *args, **kwargs: save_calls.append((args, kwargs)))

    outputs, _html_info, _html_log = postprocessing.run_postprocessing(
        1,
        None,
        [SimpleNamespace(name=str(input_file), orig_name="source.png")],
        "",
        "",
        True,
        save_format="jpg",
        skip_existing_files=True,
    )

    assert outputs == []
    assert script_calls == []
    assert save_calls == []


def test_postprocessing_saves_existing_batch_output_when_skip_disabled(monkeypatch, tmp_path):
    input_file = tmp_path / "source.png"
    output_dir = tmp_path / "outputs"
    input_file.parent.mkdir(exist_ok=True)
    output_dir.mkdir()
    Image.new("RGB", (1, 1), "red").save(input_file)
    (output_dir / "source.jpg").write_bytes(b"already here")

    script_calls = []
    save_calls = []
    setup_postprocessing_unit_test(monkeypatch, tmp_path, script_run=lambda pp, args: script_calls.append(pp))
    monkeypatch.setattr(postprocessing.images, "save_image", lambda *args, **kwargs: (save_calls.append((args, kwargs)) or (str(output_dir / "source.jpg"), None)))

    outputs, _html_info, _html_log = postprocessing.run_postprocessing(
        1,
        None,
        [SimpleNamespace(name=str(input_file), orig_name="source.png")],
        "",
        "",
        True,
        save_format="jpg",
        skip_existing_files=False,
    )

    assert len(outputs) == 1
    assert len(script_calls) == 1
    assert len(save_calls) == 1


def test_postprocessing_combines_original_parameters_with_extras_info_for_jpeg(monkeypatch, tmp_path):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    original_parameters = "a cat\nNegative prompt: blurry\nSteps: 20, Seed: 123"

    def add_upscaler_info(pp, args):
        pp.info["Postprocess upscaler"] = "ESRGAN"

    saved = {}
    setup_postprocessing_unit_test(monkeypatch, tmp_path, script_run=add_upscaler_info)
    monkeypatch.setattr(postprocessing.images, "read_info_from_image", lambda image: (original_parameters, {}))
    monkeypatch.setattr(postprocessing.images, "save_image", lambda *args, **kwargs: (saved.update(kwargs) or (str(output_dir / "source.jpg"), None)))

    postprocessing.run_postprocessing(
        0,
        Image.new("RGB", (1, 1), "red"),
        [],
        "",
        "",
        True,
        save_format="jpg",
    )

    assert saved["info"] == f"{original_parameters}\nPostprocess upscaler: ESRGAN"


def test_postprocessing_uses_preflight_png_for_skip_and_save(monkeypatch, tmp_path):
    input_file = tmp_path / "source.png"
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    Image.new("RGB", (1, 1), "red").save(input_file)
    (output_dir / "source.jpg").write_bytes(b"unrelated existing jpg")

    script_calls = []
    save_calls = []
    setup_postprocessing_unit_test(
        monkeypatch,
        tmp_path,
        script_run=lambda pp, args: script_calls.append(pp),
        output_extension=lambda args, default_extension: "png",
    )
    monkeypatch.setattr(
        postprocessing.images,
        "save_image",
        lambda *args, **kwargs: (save_calls.append(kwargs) or (str(output_dir / "source.png"), None)),
    )

    postprocessing.run_postprocessing(
        1,
        None,
        [SimpleNamespace(name=str(input_file), orig_name="source.png")],
        "",
        "",
        True,
        save_format="jpg",
        jpeg_quality=75,
        skip_existing_files=True,
    )

    assert len(script_calls) == 1
    assert len(save_calls) == 1
    assert save_calls[0]["extension"] == "png"
    assert save_calls[0]["jpeg_quality"] is None


def test_postprocessing_skips_existing_effective_png_before_scripts(monkeypatch, tmp_path):
    input_file = tmp_path / "source.png"
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    Image.new("RGB", (1, 1), "red").save(input_file)
    (output_dir / "source.png").write_bytes(b"already removed")

    script_calls = []
    setup_postprocessing_unit_test(
        monkeypatch,
        tmp_path,
        script_run=lambda pp, args: script_calls.append(pp),
        output_extension=lambda args, default_extension: "png",
    )
    save_calls = []
    monkeypatch.setattr(postprocessing.images, "save_image", lambda *args, **kwargs: save_calls.append(kwargs))

    outputs, _html_info, _html_log = postprocessing.run_postprocessing(
        1,
        None,
        [SimpleNamespace(name=str(input_file), orig_name="source.png")],
        "",
        "",
        True,
        save_format="webp",
        skip_existing_files=True,
    )

    assert outputs == []
    assert script_calls == []
    assert save_calls == []


def test_postprocessing_honors_per_image_extension_for_extra_image_suffix(monkeypatch, tmp_path):
    input_file = tmp_path / "source.png"
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    Image.new("RGB", (1, 1), "red").save(input_file)
    (output_dir / "source-alpha.png").write_bytes(b"existing transparent extra")

    def add_extra(pp, args):
        extra = pp.create_copy(Image.new("RGBA", (1, 1)), nametags=["alpha"])
        extra.output_extension = "png"
        pp.extra_images.append(extra)

    setup_postprocessing_unit_test(monkeypatch, tmp_path, script_run=add_extra)
    save_calls = []
    monkeypatch.setattr(
        postprocessing.images,
        "save_image",
        lambda *args, **kwargs: (save_calls.append(kwargs) or (str(output_dir / "source.jpg"), None)),
    )

    postprocessing.run_postprocessing(
        1,
        None,
        [SimpleNamespace(name=str(input_file), orig_name="source.png")],
        "",
        "",
        True,
        save_format="jpg",
        skip_existing_files=True,
    )

    assert [call["extension"] for call in save_calls] == ["jpg"]


def extras_api_arguments(**overrides):
    values = {
        "extras_mode": 0,
        "resize_mode": 0,
        "image": Image.new("RGB", (1, 1)),
        "image_folder": [],
        "input_dir": "",
        "output_dir": "",
        "show_extras_results": True,
        "gfpgan_visibility": 0,
        "codeformer_visibility": 0,
        "codeformer_weight": 0,
        "upscaling_resize": 2,
        "upscaling_resize_w": 2,
        "upscaling_resize_h": 2,
        "upscaling_crop": True,
        "extras_upscaler_1": "None",
        "extras_upscaler_2": "None",
        "extras_upscaler_2_visibility": 0,
        "upscale_first": False,
        "save_output": False,
    }
    values.update(overrides)
    return values


def test_extras_api_request_defaults_disable_background_removal():
    request = api_models.ExtrasSingleImageRequest()

    assert request.background_removal_enabled is False
    assert request.background_removal_model is None


def test_run_extras_maps_background_removal_api_fields_to_script_arguments(monkeypatch):
    captured = {}

    class FakeRunner:
        def create_args_for_run(self, script_args):
            captured.update(script_args)
            return []

    monkeypatch.setattr(postprocessing.scripts, "scripts_postproc", FakeRunner())
    monkeypatch.setattr(postprocessing, "run_postprocessing", lambda *args, **kwargs: ([], "", ""))

    postprocessing.run_extras(**extras_api_arguments(
        background_removal_enabled=True,
        background_removal_model="RMBG 1.4",
    ))

    assert captured["Background Removal"] == {
        "enabled": True,
        "model_name": "RMBG 1.4",
    }


def test_extras_api_translates_background_removal_errors_to_422(monkeypatch):
    request = api_models.ExtrasSingleImageRequest(
        image="encoded",
        background_removal_enabled=True,
        background_removal_model="Missing",
    )
    monkeypatch.setattr(api_module, "decode_base64_to_image", lambda image: Image.new("RGB", (1, 1)))
    monkeypatch.setattr(
        api_module.postprocessing,
        "run_extras",
        lambda **kwargs: (_ for _ in ()).throw(background_removal.BackgroundRemovalError("model is unavailable")),
    )

    with pytest.raises(HTTPException) as exc_info:
        api_module.Api.extras_single_image_api(SimpleNamespace(queue_lock=Lock()), request)

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == "model is unavailable"


def decoded_response_image(encoded):
    raw = base64.b64decode(encoded)
    return raw, Image.open(io.BytesIO(raw))


def test_enabled_extras_single_api_encodes_rgba_as_png_despite_global_jpg(monkeypatch):
    request = api_models.ExtrasSingleImageRequest(
        image="encoded",
        background_removal_enabled=True,
        background_removal_model="RMBG 1.4",
    )
    result_image = Image.new("RGBA", (2, 2), (10, 20, 30, 40))
    monkeypatch.setattr(api_module.opts, "samples_format", "jpg")
    monkeypatch.setattr(api_module, "decode_base64_to_image", lambda image: Image.new("RGB", (2, 2)))
    monkeypatch.setattr(api_module.postprocessing, "run_extras", lambda **kwargs: ([result_image], "", ""))

    response = api_module.Api.extras_single_image_api(SimpleNamespace(queue_lock=Lock()), request)
    raw, decoded = decoded_response_image(response.image)

    assert raw.startswith(b"\x89PNG\r\n\x1a\n")
    assert decoded.mode == "RGBA"


def test_enabled_extras_batch_api_encodes_rgba_as_png_despite_global_jpg(monkeypatch):
    request = api_models.ExtrasBatchImagesRequest(
        imageList=[api_models.FileData(data="encoded", name="source.png")],
        background_removal_enabled=True,
        background_removal_model="RMBG 1.4",
    )
    result_image = Image.new("RGBA", (2, 2), (10, 20, 30, 40))
    monkeypatch.setattr(api_module.opts, "samples_format", "jpg")
    monkeypatch.setattr(api_module, "decode_image_list_to_images", lambda images: [Image.new("RGB", (2, 2))])
    monkeypatch.setattr(api_module.postprocessing, "run_extras", lambda **kwargs: ([result_image], "", ""))

    response = api_module.Api.extras_batch_images_api(SimpleNamespace(queue_lock=Lock()), request)
    raw, decoded = decoded_response_image(response.images[0])

    assert raw.startswith(b"\x89PNG\r\n\x1a\n")
    assert decoded.mode == "RGBA"


def test_disabled_extras_encoding_still_uses_global_jpg(monkeypatch):
    monkeypatch.setattr(api_module.opts, "samples_format", "jpg")

    encoded = api_module.encode_pil_to_base64(Image.new("RGB", (1, 1), "red"))
    raw = base64.b64decode(encoded)

    assert raw.startswith(b"\xff\xd8\xff")
