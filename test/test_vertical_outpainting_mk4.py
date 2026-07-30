import importlib.util
import sys
import types
from pathlib import Path

import pytest
from PIL import Image


def load_mk4_module_for_test():
    import modules as modules_package

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "vertical_outpainting_mk4.py"
    stubs = {
        "gradio": types.ModuleType("gradio"),
        "modules.scripts": types.SimpleNamespace(Script=object),
        "modules.images": types.ModuleType("modules.images"),
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
        spec = importlib.util.spec_from_file_location("_test_vertical_outpainting_mk4", script_path)
        mk4 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mk4)
        return mk4
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


def test_build_placement_offsets_all_collapses_duplicates_in_order():
    mk4 = load_mk4_module_for_test()

    assert mk4.build_placement_offsets(1280, 2048, "All") == [
        ("Centered", 384),
        ("Top", 0),
        ("Bottom", 768),
        ("Top biased", 192),
        ("Bottom biased", 576),
    ]
    assert mk4.build_placement_offsets(1280, 1280, "All") == [("Centered", 0)]


def test_build_placement_offsets_explicit_and_rejects_smaller_target():
    mk4 = load_mk4_module_for_test()

    assert mk4.build_placement_offsets(1280, 2048, "Bottom biased") == [("Bottom biased", 576)]
    with pytest.raises(ValueError, match="Target height must be greater than or equal"):
        mk4.build_placement_offsets(1280, 1024, "Centered")


def test_create_target_canvas_keeps_source_width_and_pastes_at_offset():
    mk4 = load_mk4_module_for_test()
    source = Image.new("RGB", (4, 3), "red")

    canvas = mk4.create_target_canvas(source, 7, 2)

    assert canvas.size == (4, 7)
    assert canvas.getpixel((0, 0)) == (0, 0, 0)
    assert canvas.getpixel((1, 2)) == (255, 0, 0)
    assert canvas.getpixel((1, 4)) == (255, 0, 0)
    assert canvas.getpixel((1, 6)) == (0, 0, 0)


def mask_values_by_row(mask):
    return [mask.getpixel((0, y)) for y in range(mask.height)]


def test_build_source_mask_preserve_and_blend_cover_gaps_and_touching_seams():
    mk4 = load_mk4_module_for_test()

    preserve = mk4.build_source_mask((2, 4), 8, 2, "Preserve source pixels", 1)
    blend = mk4.build_source_mask((2, 4), 8, 2, "Blend seam only", 1)

    assert mask_values_by_row(preserve) == [255, 255, 255, 0, 0, 255, 255, 255]
    assert mask_values_by_row(blend) == [255, 255, 255, 0, 0, 255, 255, 255]


def test_build_source_mask_single_gap_only_adds_touching_seam():
    mk4 = load_mk4_module_for_test()

    mask = mk4.build_source_mask((2, 4), 6, 0, "Blend seam only", 1)

    assert mask_values_by_row(mask) == [0, 0, 0, 255, 255, 255]


def test_build_source_mask_soft_resynthesis_masks_entire_canvas():
    mk4 = load_mk4_module_for_test()

    mask = mk4.build_source_mask((2, 4), 8, 2, "Soft resynthesis", 1)

    assert mask_values_by_row(mask) == [255] * 8


def test_restore_source_region_preserve_pastes_entire_source():
    mk4 = load_mk4_module_for_test()
    source = Image.new("RGB", (2, 4), "red")
    generated = Image.new("RGB", (2, 8), "blue")

    restored = mk4.restore_source_region(generated, source, 2, "Preserve source pixels", 1)

    assert [restored.getpixel((0, y)) for y in range(8)] == [
        (0, 0, 255),
        (0, 0, 255),
        (255, 0, 0),
        (255, 0, 0),
        (255, 0, 0),
        (255, 0, 0),
        (0, 0, 255),
        (0, 0, 255),
    ]


def test_restore_source_region_blend_pastes_only_protected_inner_source():
    mk4 = load_mk4_module_for_test()
    source = Image.new("RGB", (2, 4), "red")
    generated = Image.new("RGB", (2, 8), "blue")

    restored = mk4.restore_source_region(generated, source, 2, "Blend seam only", 1)

    assert [restored.getpixel((0, y)) for y in range(8)] == [
        (0, 0, 255),
        (0, 0, 255),
        (0, 0, 255),
        (255, 0, 0),
        (255, 0, 0),
        (0, 0, 255),
        (0, 0, 255),
        (0, 0, 255),
    ]


def test_restore_source_region_soft_resynthesis_returns_generated_copy():
    mk4 = load_mk4_module_for_test()
    source = Image.new("RGB", (2, 4), "red")
    generated = Image.new("RGB", (2, 8), "blue")

    restored = mk4.restore_source_region(generated, source, 2, "Soft resynthesis", 1)

    assert restored.tobytes() == generated.tobytes()
    assert restored is not generated
