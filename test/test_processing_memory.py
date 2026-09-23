from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from modules import processing_output


def test_replace_saved_images_with_placeholders_replaces_only_saved_non_api_images(tmp_path):
    saved_path = str(tmp_path / "saved.png")
    original_saved = Image.new("RGB", (64, 64), "red")
    original_unsaved = Image.new("RGB", (64, 64), "blue")
    processed = SimpleNamespace(images=[original_saved, original_unsaved])
    p = SimpleNamespace(is_api=False)
    opts = SimpleNamespace(do_not_show_images=True, return_grid=False, grid_save=False, drop_saved_images_from_results=True)

    processing_output.replace_saved_images_with_placeholders(p, opts, processed, [saved_path, None])

    assert processed.images[0] is not original_saved
    assert processed.images[0].size == (1, 1)
    assert processed.images[0].already_saved_as == saved_path
    assert processed.images[1] is original_unsaved


def test_replace_saved_images_with_placeholders_keeps_api_images(tmp_path):
    saved_path = str(tmp_path / "saved.png")
    original = Image.new("RGB", (64, 64), "red")
    processed = SimpleNamespace(images=[original])
    p = SimpleNamespace(is_api=True)
    opts = SimpleNamespace(do_not_show_images=True, return_grid=False, grid_save=False, drop_saved_images_from_results=True)

    processing_output.replace_saved_images_with_placeholders(p, opts, processed, [saved_path])

    assert processed.images[0] is original


def test_replace_saved_images_with_placeholders_keeps_images_when_drop_option_disabled(tmp_path):
    saved_path = str(tmp_path / "saved.png")
    original = Image.new("RGB", (64, 64), "red")
    processed = SimpleNamespace(images=[original])
    p = SimpleNamespace(is_api=False)
    opts = SimpleNamespace(do_not_show_images=True, return_grid=False, grid_save=False, drop_saved_images_from_results=False)

    processing_output.replace_saved_images_with_placeholders(p, opts, processed, [saved_path])

    assert processed.images[0] is original


def test_replace_saved_images_with_placeholders_keeps_images_when_gallery_visible(tmp_path):
    saved_path = str(tmp_path / "saved.png")
    original = Image.new("RGB", (64, 64), "red")
    processed = SimpleNamespace(images=[original])
    p = SimpleNamespace(is_api=False)
    opts = SimpleNamespace(do_not_show_images=False, return_grid=False, grid_save=False, drop_saved_images_from_results=True)

    processing_output.replace_saved_images_with_placeholders(p, opts, processed, [saved_path])

    assert processed.images[0] is original


def test_replace_saved_images_with_placeholders_keeps_images_when_grid_enabled(tmp_path):
    saved_path = str(tmp_path / "saved.png")
    original = Image.new("RGB", (64, 64), "red")
    processed = SimpleNamespace(images=[original])
    p = SimpleNamespace(is_api=False)
    opts = SimpleNamespace(do_not_show_images=True, return_grid=True, grid_save=False, drop_saved_images_from_results=True)

    processing_output.replace_saved_images_with_placeholders(p, opts, processed, [saved_path])

    assert processed.images[0] is original


def test_replace_saved_images_with_placeholders_skips_when_script_changed_image_count(tmp_path):
    saved_path = str(tmp_path / "saved.png")
    original = Image.new("RGB", (64, 64), "red")
    processed = SimpleNamespace(images=[original])
    p = SimpleNamespace(is_api=False)
    opts = SimpleNamespace(do_not_show_images=True, return_grid=False, grid_save=False, drop_saved_images_from_results=True)

    processing_output.replace_saved_images_with_placeholders(p, opts, processed, [saved_path, None])

    assert processed.images[0] is original


def test_append_image_result_drops_saved_hidden_non_api_image(tmp_path):
    saved_path = str(tmp_path / "saved.png")
    original = Image.new("RGB", (64, 64), "red")
    p = SimpleNamespace(is_api=False)
    opts = SimpleNamespace(do_not_show_images=True, return_grid=False, grid_save=False, drop_saved_images_from_results=True)
    output_images = []
    output_image_paths = []

    processing_output.append_image_result(output_images, output_image_paths, p, opts, original, saved_path)

    assert output_images[0] is not original
    assert output_images[0].size == (1, 1)
    assert output_images[0].already_saved_as == saved_path
    assert output_image_paths == [saved_path]


def test_append_image_result_keeps_hidden_image_when_drop_option_disabled(tmp_path):
    saved_path = str(tmp_path / "saved.png")
    original = Image.new("RGB", (64, 64), "red")
    p = SimpleNamespace(is_api=False)
    opts = SimpleNamespace(do_not_show_images=True, return_grid=False, grid_save=False, drop_saved_images_from_results=False)
    output_images = []
    output_image_paths = []

    processing_output.append_image_result(output_images, output_image_paths, p, opts, original, saved_path)

    assert output_images == [original]
    assert output_image_paths == [saved_path]


def test_append_image_result_keeps_hidden_image_when_grid_needs_full_image(tmp_path):
    saved_path = str(tmp_path / "saved.png")
    original = Image.new("RGB", (64, 64), "red")
    p = SimpleNamespace(is_api=False)
    opts = SimpleNamespace(do_not_show_images=True, return_grid=True, grid_save=False, drop_saved_images_from_results=True)
    output_images = []
    output_image_paths = []

    processing_output.append_image_result(output_images, output_image_paths, p, opts, original, saved_path)

    assert output_images == [original]
    assert output_image_paths == [saved_path]


def test_append_image_result_keeps_api_image(tmp_path):
    saved_path = str(tmp_path / "saved.png")
    original = Image.new("RGB", (64, 64), "red")
    p = SimpleNamespace(is_api=True)
    opts = SimpleNamespace(do_not_show_images=True, return_grid=False, grid_save=False, drop_saved_images_from_results=True)
    output_images = []
    output_image_paths = []

    processing_output.append_image_result(output_images, output_image_paths, p, opts, original, saved_path)

    assert output_images == [original]
    assert output_image_paths == [saved_path]


def test_append_image_result_keeps_unsaved_hidden_image_even_when_drop_enabled():
    original = Image.new("RGB", (64, 64), "red")
    p = SimpleNamespace(is_api=False)
    opts = SimpleNamespace(do_not_show_images=True, return_grid=False, grid_save=False, drop_saved_images_from_results=True)
    output_images = []
    output_image_paths = []

    processing_output.append_image_result(output_images, output_image_paths, p, opts, original, None)

    assert output_images == [original]
    assert output_image_paths == [None]


def test_processing_hot_path_uses_append_image_result_for_generated_images():
    processing_source = Path("modules/processing.py").read_text()

    assert processing_source.count("processing_output.append_image_result(") >= 3
