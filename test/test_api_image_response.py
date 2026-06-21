from types import SimpleNamespace

from fastapi.exceptions import HTTPException
from PIL import Image

from modules.api import image_response


def make_processed(images):
    return SimpleNamespace(
        images=images,
        infotexts=["info"] * len(images),
        info="info",
        index_of_first_image=0,
        all_seeds=[123] * len(images),
        all_prompts=["prompt"] * len(images),
        seed=123,
        prompt="prompt",
    )


def test_normalize_api_image_return_mode_accepts_default_and_file():
    assert image_response.normalize_api_image_return_mode(None) == "base64"
    assert image_response.normalize_api_image_return_mode("BASE64") == "base64"
    assert image_response.normalize_api_image_return_mode("file") == "file"


def test_normalize_api_image_return_mode_rejects_unknown_mode():
    try:
        image_response.normalize_api_image_return_mode("stream")
    except HTTPException as exc:
        assert exc.status_code == 422
    else:
        raise AssertionError("Expected invalid image_return_mode to raise HTTPException")


def test_encode_processed_images_for_api_base64_mode_encodes_images():
    processed = make_processed([Image.new("RGB", (4, 4), "red")])

    encoded_images, image_paths = image_response.encode_processed_images_for_api(
        processed,
        True,
        "base64",
        "samples",
        "grids",
        lambda image: b"encoded",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("save_image should not be called")),
        "png",
        "png",
        False,
    )

    assert encoded_images == [b"encoded"]
    assert image_paths == []


def test_encode_processed_images_for_api_file_mode_returns_existing_paths_without_base64(tmp_path):
    saved_path = str(tmp_path / "saved.png")
    image = Image.new("RGB", (4, 4), "red")
    image.already_saved_as = saved_path
    processed = make_processed([image])

    encoded_images, image_paths = image_response.encode_processed_images_for_api(
        processed,
        True,
        "file",
        "samples",
        "grids",
        lambda image: (_ for _ in ()).throw(AssertionError("base64 encoder should not be called")),
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("save_image should not be called")),
        "png",
        "png",
        False,
    )

    assert encoded_images == []
    assert image_paths == [saved_path]


def test_encode_processed_images_for_api_file_mode_saves_unsaved_images(tmp_path):
    saved_path = str(tmp_path / "fallback.png")

    def fake_save_image(image, path, basename, seed, prompt, extension, **kwargs):
        image.already_saved_as = saved_path
        return saved_path, None

    image = Image.new("RGB", (4, 4), "red")
    processed = make_processed([image])

    encoded_images, image_paths = image_response.encode_processed_images_for_api(
        processed,
        True,
        "file",
        str(tmp_path),
        str(tmp_path),
        lambda image: (_ for _ in ()).throw(AssertionError("base64 encoder should not be called")),
        fake_save_image,
        "png",
        "png",
        False,
    )

    assert encoded_images == []
    assert image_paths == [saved_path]
    assert image.already_saved_as == saved_path


def test_encode_processed_images_for_api_send_images_false_skips_all_work():
    processed = make_processed([Image.new("RGB", (4, 4), "red")])

    encoded_images, image_paths = image_response.encode_processed_images_for_api(
        processed,
        False,
        "file",
        "samples",
        "grids",
        lambda image: (_ for _ in ()).throw(AssertionError("base64 encoder should not be called")),
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("save_image should not be called")),
        "png",
        "png",
        False,
    )

    assert encoded_images == []
    assert image_paths == []
