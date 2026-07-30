import importlib.util
import sys
import types
from pathlib import Path

import pytest
from PIL import Image


def load_mk3_module_for_test():
    import modules as modules_package

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "vertical_outpainting_mk3.py"
    stubs = {
        "gradio": types.ModuleType("gradio"),
        "modules.scripts": types.SimpleNamespace(Script=object),
        "modules.images": types.ModuleType("modules.images"),
        "modules.devices": types.SimpleNamespace(torch_gc=lambda: None),
        "modules.processing": types.SimpleNamespace(Processed=object, process_images=lambda *args, **kwargs: None),
        "modules.shared": types.SimpleNamespace(opts=types.SimpleNamespace(), state=types.SimpleNamespace()),
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
        spec = importlib.util.spec_from_file_location("_test_vertical_outpainting_mk3", script_path)
        mk3 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mk3)
        return mk3
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


def test_build_shift_sequences_all_with_both_directions():
    mk3 = load_mk3_module_for_test()

    assert mk3.build_shift_sequences(["up", "down"], "All", 384) == [
        ("Centered", [("up", 384), ("down", 384)]),
        ("Top", [("up", 384), ("up", 384)]),
        ("Bottom", [("down", 384), ("down", 384)]),
        ("Top Centered", [("up", 384), ("down", 192), ("up", 192)]),
        ("Bottom Centered", [("down", 384), ("up", 192), ("down", 192)]),
    ]


def test_build_shift_sequences_filters_single_direction_presets():
    mk3 = load_mk3_module_for_test()

    assert mk3.build_shift_sequences(["up"], "All", 256) == [
        ("Top", [("up", 256), ("up", 256)]),
    ]
    assert mk3.build_shift_sequences(["down"], "All", 256) == [
        ("Bottom", [("down", 256), ("down", 256)]),
    ]
    assert mk3.build_shift_sequences(["up"], "Centered", 256) == []


def test_build_shift_sequences_respects_explicit_preset():
    mk3 = load_mk3_module_for_test()

    assert mk3.build_shift_sequences(["up", "down"], "Top Centered", 128) == [
        ("Top Centered", [("up", 128), ("down", 64), ("up", 64)]),
    ]
    assert mk3.build_shift_sequences([], "All", 128) == []


def test_zoom_content_centered_preserves_size_and_center_pixel():
    mk3 = load_mk3_module_for_test()
    image = Image.new("RGB", (20, 20), "black")
    pixels = image.load()
    for y in range(20):
        for x in range(20):
            if x < 10 and y < 10:
                pixels[x, y] = (255, 0, 0)
            elif x >= 10 and y < 10:
                pixels[x, y] = (0, 255, 0)
            elif x < 10 and y >= 10:
                pixels[x, y] = (0, 0, 255)
            else:
                pixels[x, y] = (255, 255, 255)

    zoomed = mk3.zoom_content_centered(image, 2.0)

    assert zoomed.size == image.size
    assert zoomed.getpixel((0, 0)) == (255, 0, 0)
    assert zoomed.getpixel((19, 0)) == (0, 255, 0)
    assert zoomed.getpixel((0, 19)) == (0, 0, 255)
    assert zoomed.getpixel((19, 19)) == (255, 255, 255)
    assert zoomed.tobytes() != image.tobytes()


@pytest.mark.parametrize(("expand_pixels", "pass_direction"), [(8, "up"), (72, "down")])
def test_expand_vertical_once_returns_exact_requested_canvas(expand_pixels, pass_direction):
    mk3 = load_mk3_module_for_test()
    init_img = Image.new("RGB", (101, 99), "red")

    def split_grid(image, **_kwargs):
        return types.SimpleNamespace(tiles=[], image=image)

    mk3.images.split_grid = split_grid
    mk3.images.combine_grid = lambda grid: grid.image

    result, _seed, _info = mk3._expand_vertical_once(
        types.SimpleNamespace(width=128, height=128),
        init_img,
        expand_pixels,
        pass_direction,
        4,
    )

    assert result.size == (101, 99 + expand_pixels)
    original_top = expand_pixels if pass_direction == "up" else 0
    assert result.crop((0, original_top, 101, original_top + 99)).tobytes() == init_img.tobytes()


def test_safe_overlap_stays_strictly_below_each_tile_dimension():
    mk3 = load_mk3_module_for_test()

    assert mk3._safe_overlap(384, 512, 384) == 383
    assert mk3._safe_overlap(999, 256, 128) == 127
    assert mk3._safe_overlap(0, 256, 128) == 0


def test_run_uses_each_preset_representative_metadata_for_saved_and_returned_images():
    mk3 = load_mk3_module_for_test()
    saved_images = []
    pass_results = iter([
        (101, "info-101"), (102, "info-102"),
        (201, "info-201"), (202, "info-202"),
        (301, "info-301"), (302, "info-302"),
        (401, "info-401"), (402, "info-402"), (403, "info-403"),
        (501, "info-501"), (502, "info-502"), (503, "info-503"),
    ])

    class FakeProcessed:
        def __init__(self, _p, images, seed, info, **kwargs):
            self.images = images
            self.seed = seed
            self.info = info
            self.all_seeds = kwargs.get("all_seeds")
            self.infotexts = kwargs.get("infotexts")

    def expand_once(_p, image, _pixels, _direction, _mask_blur):
        seed, info = next(pass_results)
        return image, seed, info

    mk3.Processed = FakeProcessed
    mk3._expand_vertical_once = expand_once
    mk3.opts = types.SimpleNamespace(samples_save=True, samples_format="png")
    mk3.images.save_image = lambda image, _path, _basename, seed, _prompt, _format, info, p: saved_images.append((image, seed, info, p))
    mk3.state = types.SimpleNamespace()
    p = types.SimpleNamespace(
        prompt="base prompt",
        init_images=[Image.new("RGB", (16, 16), "blue")],
        n_iter=2,
        batch_size=2,
        do_not_save_grid=False,
        do_not_save_samples=False,
        mask_blur=0,
        inpainting_fill=0,
        inpaint_full_res=True,
        extra_generation_params={},
        outpath_samples="samples",
        seed=42,
    )

    result = mk3.Script().run(p, 8, 4, 0, ["up", "down"], "All", 1.0, False, "")

    assert result.seed == 101
    assert result.info == "info-101"
    assert result.all_seeds == [101, 201, 301, 401, 501]
    assert result.infotexts == ["info-101", "info-201", "info-301", "info-401", "info-501"]
    assert [(seed, info) for _image, seed, info, _p in saved_images] == list(zip(result.all_seeds, result.infotexts))


def test_run_distinguishes_invalid_preset_from_missing_direction():
    mk3 = load_mk3_module_for_test()

    class FakeProcessed:
        def __init__(self, _p, _images, _seed, info, **_kwargs):
            self.info = info

    mk3.Processed = FakeProcessed
    p = types.SimpleNamespace(
        prompt="base prompt",
        init_images=[Image.new("RGB", (16, 16), "blue")],
        n_iter=1,
        batch_size=1,
        do_not_save_grid=False,
        do_not_save_samples=False,
        mask_blur=0,
        inpainting_fill=0,
        inpaint_full_res=False,
        seed=42,
    )

    no_direction = mk3.Script().run(p, 8, 4, 0, [], "All", 1.0, False, "")
    invalid_preset = mk3.Script().run(p, 8, 4, 0, ["up"], "Centered", 1.0, False, "")

    assert no_direction.info == "No vertical outpainting direction selected."
    assert invalid_preset.info == "Selected shift preset is not available for the chosen outpainting directions."
