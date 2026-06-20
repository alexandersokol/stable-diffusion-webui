from PIL import Image

from modules import progress_results


def test_lightweight_result_preserves_tuple_shape_and_non_gallery_outputs():
    result = ([Image.new("RGB", (64, 64), "red")], "generation info", "<p>info</p>", "<p>log</p>")

    lightweight = progress_results.create_lightweight_recorded_result(result, max_image_size=32)

    assert isinstance(lightweight, tuple)
    assert lightweight[1:] == result[1:]


def test_lightweight_result_replaces_saved_image_with_placeholder(tmp_path):
    image = Image.new("RGB", (1024, 1024), "red")
    saved_path = str(tmp_path / "saved.png")
    image.already_saved_as = saved_path

    lightweight = progress_results.create_lightweight_recorded_result(([image], "info"), max_image_size=256)
    restored_image = lightweight[0][0]

    assert restored_image is not image
    assert restored_image.size == (1, 1)
    assert restored_image.already_saved_as == saved_path


def test_lightweight_result_downscales_unsaved_large_image():
    image = Image.new("RGB", (2048, 1024), "blue")
    image.info["parameters"] = "prompt"

    lightweight = progress_results.create_lightweight_recorded_result(([image], "info"), max_image_size=512)
    restored_image = lightweight[0][0]

    assert restored_image is not image
    assert restored_image.size == (512, 256)
    assert restored_image.info["parameters"] == "prompt"


def test_lightweight_result_copies_unsaved_image_when_resize_disabled():
    image = Image.new("RGB", (2048, 1024), "blue")

    lightweight = progress_results.create_lightweight_recorded_result(([image], "info"), max_image_size=0)
    restored_image = lightweight[0][0]

    assert restored_image is not image
    assert restored_image.size == image.size


def test_lightweight_result_leaves_non_gallery_first_output_unchanged():
    result = ("not a gallery", "info")

    assert progress_results.create_lightweight_recorded_result(result) == result
