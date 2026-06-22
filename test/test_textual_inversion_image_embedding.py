import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageFont

from modules.textual_inversion import image_embedding


REFERENCE_RANDOM = [
    253, 242, 127, 44, 157, 27, 239, 133, 38, 79, 167, 4, 177,
    95, 130, 79, 78, 14, 52, 215, 220, 194, 126, 28, 240, 179,
    160, 153, 149, 50, 105, 14, 21, 218, 199, 18, 54, 198, 193,
    38, 128, 19, 53, 195, 124, 75, 205, 12, 6, 145, 0, 28,
    30, 148, 8, 45, 218, 171, 55, 249, 97, 166, 12, 35, 0,
    41, 221, 122, 215, 170, 31, 113, 186, 97, 119, 31, 23, 185,
    66, 140, 30, 41, 37, 63, 137, 109, 216, 55, 159, 145, 82,
    204, 86, 73, 222, 44, 198, 118, 240, 97,
]


def legacy_xor_block(block):
    g = image_embedding.lcg()
    randblock = np.array([next(g) for _ in range(np.prod(block.shape))]).astype(np.uint8).reshape(block.shape)
    return np.bitwise_xor(block.astype(np.uint8), randblock & 0x0F)


def expected_caption_alpha(height):
    from math import cos

    factor = 1.5
    values = []
    for y in range(height):
        mag = 1 - cos(y / height * factor)
        mag = max(mag, 1 - cos((height - y) / height * factor * 1.1))
        values.append(int(mag * 255))

    return np.array([min(value, 255) for value in values], dtype=np.uint8)


def test_lcg_bytes_matches_legacy_sequence():
    assert image_embedding.lcg_bytes(100).tolist() == REFERENCE_RANDOM


def test_xor_block_matches_legacy_generator_output():
    block = (np.arange(7 * 5 * 3, dtype=np.uint8).reshape((7, 5, 3)) * 3) % 251

    assert np.array_equal(image_embedding.xor_block(block), legacy_xor_block(block))


def test_caption_image_overlay_builds_gradient_without_per_pixel_writes(monkeypatch):
    image = Image.new("RGBA", (192, 48), (255, 255, 255, 255))
    fake_images = SimpleNamespace(get_font=lambda size: ImageFont.load_default())
    monkeypatch.setitem(sys.modules, "modules.images", fake_images)

    def fail_putpixel(*args, **kwargs):
        raise AssertionError("caption gradient should be built as an array, not with per-pixel putpixel calls")

    monkeypatch.setattr(Image.Image, "putpixel", fail_putpixel)

    overlay = image_embedding.caption_image_overlay(image, "title", "left", "middle", "right")

    expected_red = 255 - expected_caption_alpha(image.height)
    assert np.array_equal(np.asarray(overlay.getchannel("R"))[:, 0], expected_red)
