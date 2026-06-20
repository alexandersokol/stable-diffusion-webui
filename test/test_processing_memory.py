from types import SimpleNamespace

from PIL import Image

from modules import processing_output


def test_replace_saved_images_with_placeholders_replaces_only_saved_non_api_images(tmp_path):
    saved_path = str(tmp_path / "saved.png")
    original_saved = Image.new("RGB", (64, 64), "red")
    original_unsaved = Image.new("RGB", (64, 64), "blue")
    processed = SimpleNamespace(images=[original_saved, original_unsaved])
    p = SimpleNamespace(is_api=False)

    processing_output.replace_saved_images_with_placeholders(p, processed, [saved_path, None])

    assert processed.images[0] is not original_saved
    assert processed.images[0].size == (1, 1)
    assert processed.images[0].already_saved_as == saved_path
    assert processed.images[1] is original_unsaved


def test_replace_saved_images_with_placeholders_keeps_api_images(tmp_path):
    saved_path = str(tmp_path / "saved.png")
    original = Image.new("RGB", (64, 64), "red")
    processed = SimpleNamespace(images=[original])
    p = SimpleNamespace(is_api=True)

    processing_output.replace_saved_images_with_placeholders(p, processed, [saved_path])

    assert processed.images[0] is original


def test_replace_saved_images_with_placeholders_skips_when_script_changed_image_count(tmp_path):
    saved_path = str(tmp_path / "saved.png")
    original = Image.new("RGB", (64, 64), "red")
    processed = SimpleNamespace(images=[original])
    p = SimpleNamespace(is_api=False)

    processing_output.replace_saved_images_with_placeholders(p, processed, [saved_path, None])

    assert processed.images[0] is original
