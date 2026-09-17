from types import SimpleNamespace

import requests
from PIL import Image

from modules import postprocessing


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


def setup_postprocessing_unit_test(monkeypatch, tmp_path, *, script_run):
    opts = SimpleNamespace(
        enable_pnginfo=False,
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
    monkeypatch.setattr(postprocessing.scripts, "scripts_postproc", SimpleNamespace(run=script_run))


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
