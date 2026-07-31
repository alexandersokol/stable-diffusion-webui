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


def test_restore_source_region_preserve_with_soft_blend_feathers_edges_only():
    mk4 = load_mk4_module_for_test()
    source = Image.new("RGB", (1, 6), "red")
    generated = Image.new("RGB", (1, 10), "blue")

    restored = mk4.restore_source_region(generated, source, 2, "Preserve source pixels", 1, 2)

    assert restored.getpixel((0, 0)) == (0, 0, 255)
    assert restored.getpixel((0, 9)) == (0, 0, 255)
    assert restored.getpixel((0, 2)) not in [(255, 0, 0), (0, 0, 255)]
    assert restored.getpixel((0, 3)) not in [(255, 0, 0), (0, 0, 255)]
    assert restored.getpixel((0, 4)) == (255, 0, 0)
    assert restored.getpixel((0, 5)) == (255, 0, 0)
    assert restored.getpixel((0, 6)) not in [(255, 0, 0), (0, 0, 255)]
    assert restored.getpixel((0, 7)) not in [(255, 0, 0), (0, 0, 255)]


def test_restore_source_region_soft_resynthesis_ignores_soft_blend():
    mk4 = load_mk4_module_for_test()
    source = Image.new("RGB", (1, 4), "red")
    generated = Image.new("RGB", (1, 8), "blue")

    restored = mk4.restore_source_region(generated, source, 2, "Soft resynthesis", 1, 2)

    assert restored.tobytes() == generated.tobytes()


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


def test_restore_source_region_blend_soft_blend_preserves_generated_rows_outside_band():
    mk4 = load_mk4_module_for_test()
    source = Image.new("RGB", (1, 10), "red")
    generated = Image.new("RGB", (1, 14), "blue")

    restored = mk4.restore_source_region(generated, source, 2, "Blend seam only", 4, 2)

    assert restored.getpixel((0, 2)) not in [(255, 0, 0), (0, 0, 255)]
    assert restored.getpixel((0, 3)) not in [(255, 0, 0), (0, 0, 255)]
    assert restored.getpixel((0, 4)) == (0, 0, 255)
    assert restored.getpixel((0, 5)) == (0, 0, 255)
    assert restored.getpixel((0, 6)) == (255, 0, 0)
    assert restored.getpixel((0, 7)) == (255, 0, 0)
    assert restored.getpixel((0, 8)) == (0, 0, 255)
    assert restored.getpixel((0, 9)) == (0, 0, 255)
    assert restored.getpixel((0, 10)) not in [(255, 0, 0), (0, 0, 255)]
    assert restored.getpixel((0, 11)) not in [(255, 0, 0), (0, 0, 255)]


def test_restore_source_region_soft_resynthesis_returns_generated_copy():
    mk4 = load_mk4_module_for_test()
    source = Image.new("RGB", (2, 4), "red")
    generated = Image.new("RGB", (2, 8), "blue")

    restored = mk4.restore_source_region(generated, source, 2, "Soft resynthesis", 1)

    assert restored.tobytes() == generated.tobytes()
    assert restored is not generated


class FakeProcessed:
    def __init__(self, _p, images, seed, info, **kwargs):
        self.images = images
        self.seed = seed
        self.info = info
        self.all_seeds = kwargs.get("all_seeds")
        self.infotexts = kwargs.get("infotexts")


def make_p(source):
    return types.SimpleNamespace(
        prompt="base prompt",
        init_images=[source],
        image_mask=None,
        width=64,
        height=64,
        n_iter=2,
        batch_size=3,
        do_not_save_grid=False,
        do_not_save_samples=False,
        mask_blur=9,
        inpainting_fill=2,
        extra_generation_params={},
        outpath_samples="samples",
        seed=42,
    )


def test_run_generates_all_final_placements_and_saves_without_grid():
    mk4 = load_mk4_module_for_test()
    calls = []
    saved = []

    def fake_process(p):
        calls.append((p.init_images[0].size, p.image_mask.size, p.width, p.height, p.do_not_save_grid))
        seed = 100 + len(calls)
        color = (seed, 0, 0)
        return FakeProcessed(p, [Image.new("RGB", (p.width, p.height), color)], seed, f"info-{seed}")

    mk4.Processed = FakeProcessed
    mk4.process_images = fake_process
    mk4.opts = types.SimpleNamespace(samples_save=True, samples_format="png")
    mk4.images.save_image = lambda image, _path, _basename, seed, _prompt, _format, info, p: saved.append((image.size, seed, info, p.do_not_save_grid))
    mk4.state = types.SimpleNamespace(job="", job_count=0)
    p = make_p(Image.new("RGB", (4, 4), "red"))

    result = mk4.Script().run(p, 8, "All", "Preserve source pixels", 1, 4, 0, 0, False, "")

    assert len(result.images) == 5
    assert result.all_seeds == [101, 102, 103, 104, 105]
    assert result.infotexts == ["info-101", "info-102", "info-103", "info-104", "info-105"]
    assert calls == [((4, 8), (4, 8), 4, 8, True)] * 5
    assert saved == [((4, 8), seed, f"info-{seed}", True) for seed in result.all_seeds]


def test_run_preserves_source_pixels_or_leaves_soft_resynthesis_generated():
    mk4 = load_mk4_module_for_test()

    def fake_process(p):
        return FakeProcessed(p, [Image.new("RGB", (p.width, p.height), "blue")], 101, "info-101")

    mk4.Processed = FakeProcessed
    mk4.process_images = fake_process
    mk4.opts = types.SimpleNamespace(samples_save=False, samples_format="png")
    mk4.state = types.SimpleNamespace(job="", job_count=0)
    source = Image.new("RGB", (2, 4), "red")

    preserve = mk4.Script().run(make_p(source), 8, "Centered", "Preserve source pixels", 1, 4, 0, 0, False, "")
    soft = mk4.Script().run(make_p(source), 8, "Centered", "Soft resynthesis", 1, 4, 0, 0, False, "")

    assert preserve.images[0].getpixel((0, 2)) == (255, 0, 0)
    assert preserve.images[0].getpixel((0, 3)) == (255, 0, 0)
    assert soft.images[0].getpixel((0, 2)) == (0, 0, 255)


def test_run_soft_resynthesis_forces_original_masked_content():
    mk4 = load_mk4_module_for_test()
    inpainting_fill_values = []

    def fake_process(p):
        inpainting_fill_values.append(p.inpainting_fill)
        return FakeProcessed(p, [Image.new("RGB", (p.width, p.height), "blue")], 101, "info-101")

    mk4.Processed = FakeProcessed
    mk4.process_images = fake_process
    mk4.opts = types.SimpleNamespace(samples_save=False, samples_format="png")
    mk4.state = types.SimpleNamespace(job="", job_count=0)

    mk4.Script().run(make_p(Image.new("RGB", (2, 4), "red")), 8, "Centered", "Soft resynthesis", 1, 4, 0, 3, False, "")

    assert inpainting_fill_values == [mk4.SOFT_RESYNTHESIS_INPAINTING_FILL]


def test_run_injects_continue_prompt_and_restores_processing_state():
    mk4 = load_mk4_module_for_test()
    prompts_seen = []

    def fake_process(p):
        prompts_seen.append(p.prompt)
        return FakeProcessed(p, [Image.new("RGB", (p.width, p.height), "blue")], 101, "info-101")

    mk4.Processed = FakeProcessed
    mk4.process_images = fake_process
    mk4.opts = types.SimpleNamespace(samples_save=False, samples_format="png")
    mk4.state = types.SimpleNamespace(job="", job_count=0)
    p = make_p(Image.new("RGB", (2, 4), "red"))
    original_init_images = p.init_images

    mk4.Script().run(p, 8, "Centered", "Preserve source pixels", 1, 4, 0, 0, True, "continue naturally")

    assert prompts_seen == ["base prompt, continue naturally"]
    assert p.prompt == "base prompt"
    assert p.init_images is original_init_images
    assert p.image_mask is None
    assert p.width == 64
    assert p.height == 64
    assert p.n_iter == 2
    assert p.batch_size == 3
    assert p.do_not_save_grid is False
    assert p.do_not_save_samples is False
    assert p.mask_blur == 9
    assert p.inpainting_fill == 2


def test_run_returns_clear_errors_for_missing_source_and_small_target():
    mk4 = load_mk4_module_for_test()
    mk4.Processed = FakeProcessed

    missing = make_p(Image.new("RGB", (2, 4), "red"))
    missing.init_images = []
    missing_result = mk4.Script().run(missing, 8, "Centered", "Preserve source pixels", 1, 4, 0, 0, False, "")

    small = make_p(Image.new("RGB", (2, 4), "red"))
    small_result = mk4.Script().run(small, 2, "Centered", "Preserve source pixels", 1, 4, 0, 0, False, "")

    assert missing_result.info == "Vertical Outpainting Mk4 requires one source image."
    assert small_result.info == "Target height must be greater than or equal to the source image height."


def test_run_records_mk4_generation_metadata_and_respects_explicit_placement():
    mk4 = load_mk4_module_for_test()
    masks = []

    def fake_process(p):
        masks.append(p.image_mask.copy())
        return FakeProcessed(p, [Image.new("RGB", (p.width, p.height), "blue")], 101, "info-101")

    mk4.Processed = FakeProcessed
    mk4.process_images = fake_process
    mk4.opts = types.SimpleNamespace(samples_save=False, samples_format="png")
    mk4.state = types.SimpleNamespace(job="", job_count=0)
    p = make_p(Image.new("RGB", (2, 4), "red"))

    result = mk4.Script().run(p, 8, "Bottom", "Blend seam only", 1, 4, 32, 0, True, "continue naturally")

    assert len(result.images) == 1
    assert mask_values_by_row(masks[0]) == [255, 255, 255, 255, 255, 0, 0, 0]
    assert p.extra_generation_params["Vertical Outpainting MK4 target height"] == 8
    assert p.extra_generation_params["Vertical Outpainting MK4 source placement"] == "Bottom"
    assert p.extra_generation_params["Vertical Outpainting MK4 source handling"] == "Blend seam only"
    assert p.extra_generation_params["Vertical Outpainting MK4 seam size"] == 1
    assert p.extra_generation_params["Vertical Outpainting MK4 mask blur"] == 4
    assert p.extra_generation_params["Vertical Outpainting MK4 soft seam blend"] == 32
    assert p.extra_generation_params["Vertical Outpainting MK4 continue prompt injected"] is True


def test_seam_size_is_clamped_to_source_height():
    mk4 = load_mk4_module_for_test()

    mask = mk4.build_source_mask((2, 4), 8, 2, "Blend seam only", 999)
    restored = mk4.restore_source_region(Image.new("RGB", (2, 8), "blue"), Image.new("RGB", (2, 4), "red"), 2, "Blend seam only", 999)

    assert mask_values_by_row(mask) == [255] * 8
    assert restored.tobytes() == Image.new("RGB", (2, 8), "blue").tobytes()


def test_run_forces_normal_mask_polarity_and_restores_inpaint_settings():
    mk4 = load_mk4_module_for_test()
    settings_seen = []

    def fake_process(p):
        settings_seen.append((p.inpainting_mask_invert, p.inpaint_full_res))
        return FakeProcessed(p, [Image.new("RGB", (p.width, p.height), "blue")], 101, "info-101")

    mk4.Processed = FakeProcessed
    mk4.process_images = fake_process
    mk4.opts = types.SimpleNamespace(samples_save=False, samples_format="png")
    mk4.state = types.SimpleNamespace(job="", job_count=0)
    p = make_p(Image.new("RGB", (2, 4), "red"))
    p.inpainting_mask_invert = 1
    p.inpaint_full_res = True

    mk4.Script().run(p, 8, "Centered", "Preserve source pixels", 1, 4, 0, 0, False, "")

    assert settings_seen == [(0, True)]
    assert p.inpainting_mask_invert == 1
    assert p.inpaint_full_res is True


def test_run_stops_on_empty_processed_images_without_placeholder_or_save():
    mk4 = load_mk4_module_for_test()
    saved = []
    calls = []

    def fake_process(p):
        calls.append(p.seed)
        if len(calls) == 1:
            return FakeProcessed(p, [Image.new("RGB", (p.width, p.height), "blue")], 101, "info-101")
        return FakeProcessed(p, [], 102, "info-102")

    mk4.Processed = FakeProcessed
    mk4.process_images = fake_process
    mk4.opts = types.SimpleNamespace(samples_save=True, samples_format="png", save_incomplete_images=False)
    mk4.images.save_image = lambda _image, _path, _basename, seed, _prompt, _format, **kwargs: saved.append((seed, kwargs["info"]))
    mk4.state = types.SimpleNamespace(job="", job_count=0, interrupted=False, skipped=False)
    p = make_p(Image.new("RGB", (2, 4), "red"))

    result = mk4.Script().run(p, 8, "All", "Preserve source pixels", 1, 4, 0, 0, False, "")

    assert calls == [42, 102]
    assert len(result.images) == 1
    assert result.all_seeds == [101]
    assert result.infotexts == ["info-101"]
    assert saved == [(101, "info-101")]


def test_run_returns_empty_result_without_saving_when_first_generation_is_empty():
    mk4 = load_mk4_module_for_test()
    saved = []

    mk4.Processed = FakeProcessed
    mk4.process_images = lambda p: FakeProcessed(p, [], 101, "info-101")
    mk4.opts = types.SimpleNamespace(samples_save=True, samples_format="png", save_incomplete_images=False)
    mk4.images.save_image = lambda *args, **kwargs: saved.append(args)
    mk4.state = types.SimpleNamespace(job="", job_count=0, interrupted=False, skipped=False)
    p = make_p(Image.new("RGB", (2, 4), "red"))

    result = mk4.Script().run(p, 8, "Centered", "Preserve source pixels", 1, 4, 0, 0, False, "")

    assert result.images == []
    assert result.all_seeds == []
    assert result.infotexts == []
    assert saved == []


def test_run_does_not_save_when_original_request_disables_sample_saving():
    mk4 = load_mk4_module_for_test()
    saved = []

    mk4.Processed = FakeProcessed
    mk4.process_images = lambda p: FakeProcessed(p, [Image.new("RGB", (p.width, p.height), "blue")], 101, "info-101")
    mk4.opts = types.SimpleNamespace(samples_save=True, samples_format="png", save_incomplete_images=False)
    mk4.images.save_image = lambda *args, **kwargs: saved.append(args)
    mk4.state = types.SimpleNamespace(job="", job_count=0, interrupted=False, skipped=False)
    p = make_p(Image.new("RGB", (2, 4), "red"))
    p.do_not_save_samples = True

    mk4.Script().run(p, 8, "Centered", "Preserve source pixels", 1, 4, 0, 0, False, "")

    assert saved == []


def test_run_does_not_save_interrupted_results_unless_incomplete_saves_are_enabled():
    mk4 = load_mk4_module_for_test()
    saved = []

    mk4.Processed = FakeProcessed
    mk4.process_images = lambda p: FakeProcessed(p, [Image.new("RGB", (p.width, p.height), "blue")], 101, "info-101")
    mk4.opts = types.SimpleNamespace(samples_save=True, samples_format="png", save_incomplete_images=False)
    mk4.images.save_image = lambda *args, **kwargs: saved.append(args)
    mk4.state = types.SimpleNamespace(job="", job_count=0, interrupted=True, skipped=False)
    p = make_p(Image.new("RGB", (2, 4), "red"))

    mk4.Script().run(p, 8, "Centered", "Preserve source pixels", 1, 4, 0, 0, False, "")

    assert saved == []
