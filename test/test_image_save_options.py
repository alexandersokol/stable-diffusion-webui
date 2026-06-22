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
