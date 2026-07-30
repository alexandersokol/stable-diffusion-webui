import importlib.util
import sys
import types
from pathlib import Path

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
