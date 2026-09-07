from types import SimpleNamespace
import sys
import types

from PIL import Image

sys.modules.setdefault("modules.sd_samplers", types.ModuleType("modules.sd_samplers"))

from modules import images


def test_png_save_uses_configured_compress_level(monkeypatch, tmp_path):
    recorded = {}

    def fake_save(self, filename, **kwargs):
        recorded.update(kwargs)
        with open(filename, "wb") as file:
            file.write(b"png")

    monkeypatch.setattr(Image.Image, "save", fake_save)
    monkeypatch.setattr(images, "opts", SimpleNamespace(enable_pnginfo=False, jpeg_quality=80, webp_lossless=False, png_compress_level=1))

    images.save_image_with_geninfo(Image.new("RGB", (1, 1)), "info", str(tmp_path / "image.png"))

    assert recorded["compress_level"] == 1


def test_jpeg_save_can_use_per_save_quality_override(monkeypatch, tmp_path):
    recorded = {}

    def fake_save(self, filename, **kwargs):
        recorded.update(kwargs)
        with open(filename, "wb") as file:
            file.write(b"jpg")

    monkeypatch.setattr(Image.Image, "save", fake_save)
    monkeypatch.setattr(images, "opts", SimpleNamespace(enable_pnginfo=False, jpeg_quality=80, webp_lossless=False, png_compress_level=1))

    images.save_image_with_geninfo(Image.new("RGB", (1, 1)), "info", str(tmp_path / "image.jpg"), jpeg_quality=90)

    assert recorded["quality"] == 90


def test_save_image_forwards_per_save_quality_override(monkeypatch, tmp_path):
    recorded = {}

    class FakeImageSaveParams:
        def __init__(self, image, p, filename, pnginfo):
            self.image = image
            self.p = p
            self.filename = filename
            self.pnginfo = pnginfo

    def fake_save_image_with_geninfo(image, geninfo, filename, extension=None, existing_pnginfo=None, pnginfo_section_name="parameters", jpeg_quality=None):
        recorded["jpeg_quality"] = jpeg_quality
        with open(filename, "wb") as file:
            file.write(b"jpg")

    save_opts = SimpleNamespace(
        directories_filename_pattern="",
        export_for_4chan=False,
        grid_save_to_dirs=False,
        img_downscale_threshold=4.0,
        save_images_add_number=False,
        save_images_replace_action="Replace",
        save_to_dirs=False,
        save_txt=False,
        samples_filename_pattern="",
        target_side_length=4000,
    )
    monkeypatch.setattr(images, "opts", save_opts)
    monkeypatch.setattr(images.shared, "opts", save_opts)
    monkeypatch.setattr(images.script_callbacks, "ImageSaveParams", FakeImageSaveParams)
    monkeypatch.setattr(images.script_callbacks, "before_image_saved_callback", lambda params: None)
    monkeypatch.setattr(images.script_callbacks, "image_saved_callback", lambda params: None)
    monkeypatch.setattr(images, "save_image_with_geninfo", fake_save_image_with_geninfo)

    images.save_image(
        Image.new("RGB", (1, 1)),
        path=str(tmp_path),
        basename="",
        extension="jpg",
        info="info",
        short_filename=True,
        no_prompt=True,
        forced_filename="image",
        jpeg_quality=90,
    )

    assert recorded["jpeg_quality"] == 90
