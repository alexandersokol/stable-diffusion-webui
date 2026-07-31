import importlib.util
import sys
import types
from pathlib import Path

import pytest
from PIL import Image


def load_mk5_module_for_test():
    import modules as modules_package

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "vertical_outpainting_mk5.py"

    def fake_create_infotext(p, all_prompts, all_seeds, all_subseeds, comments=None, all_negative_prompts=None):
        extra_params = ", ".join(f"{key}: {value}" for key, value in p.extra_generation_params.items())
        return (
            f"{all_prompts[0]}\n"
            f"Negative prompt: {all_negative_prompts[0]}\n"
            f"Seed: {all_seeds[0]}, Variation seed: {all_subseeds[0]}, "
            f"Size: {p.width}x{p.height}, {extra_params}"
        )

    stubs = {
        "gradio": types.ModuleType("gradio"),
        "modules.scripts": types.SimpleNamespace(Script=object),
        "modules.images": types.ModuleType("modules.images"),
        "modules.processing": types.SimpleNamespace(
            Processed=object,
            create_infotext=fake_create_infotext,
            process_images=lambda *args, **kwargs: None,
        ),
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
        spec = importlib.util.spec_from_file_location("_test_vertical_outpainting_mk5", script_path)
        mk5 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mk5)
        return mk5
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


def test_compute_scale_factor_interpolates_between_width_fit_and_height_fit():
    mk5 = load_mk5_module_for_test()

    assert mk5.compute_scale_factor(1280, 2048, 0.0) == pytest.approx(1.0)
    assert mk5.compute_scale_factor(1280, 2048, 0.5) == pytest.approx(1.3)
    assert mk5.compute_scale_factor(1280, 2048, 1.0) == pytest.approx(1.6)


def test_prepare_scaled_source_preserves_target_width_and_crops_center():
    mk5 = load_mk5_module_for_test()
    source = Image.new("RGB", (10, 10), "black")
    pixels = source.load()
    for y in range(10):
        for x in range(10):
            pixels[x, y] = (x * 20, y * 20, 0)

    prepared = mk5.prepare_scaled_source(source, 20, 1.0)

    assert prepared.size == (10, 20)
    assert prepared.getpixel((0, 10)) != prepared.getpixel((9, 10))


def test_compute_vertical_offset_for_each_placement():
    mk5 = load_mk5_module_for_test()

    assert mk5.compute_vertical_offset(1280, 2048, "Top") == 0
    assert mk5.compute_vertical_offset(1280, 2048, "Bottom") == 768
    assert mk5.compute_vertical_offset(1280, 2048, "Center") == 384


def test_create_target_canvas_pastes_prepared_source_at_offset():
    mk5 = load_mk5_module_for_test()
    prepared = Image.new("RGB", (4, 3), "red")

    canvas = mk5.create_target_canvas(prepared, 7, 2)

    assert canvas.size == (4, 7)
    assert canvas.getpixel((0, 0)) == (0, 0, 0)
    assert canvas.getpixel((1, 2)) == (255, 0, 0)
    assert canvas.getpixel((1, 4)) == (255, 0, 0)
    assert canvas.getpixel((1, 6)) == (0, 0, 0)


def mask_values_by_row(mask):
    return [mask.getpixel((0, y)) for y in range(mask.height)]


def test_build_outpaint_mask_centered_top_bottom_and_no_gap():
    mk5 = load_mk5_module_for_test()

    centered = mk5.build_outpaint_mask((2, 4), 8, 2, 1)
    top = mk5.build_outpaint_mask((2, 4), 8, 0, 1)
    bottom = mk5.build_outpaint_mask((2, 4), 8, 4, 1)
    no_gap = mk5.build_outpaint_mask((2, 8), 8, 0, 1)

    assert mask_values_by_row(centered) == [255, 255, 255, 0, 0, 255, 255, 255]
    assert mask_values_by_row(top) == [0, 0, 0, 255, 255, 255, 255, 255]
    assert mask_values_by_row(bottom) == [255, 255, 255, 255, 255, 0, 0, 0]
    assert mask_values_by_row(no_gap) == [0] * 8
    assert mk5.mask_has_white(centered) is True
    assert mk5.mask_has_white(no_gap) is False


def test_restore_prepared_source_pastes_exact_pixels():
    mk5 = load_mk5_module_for_test()
    prepared = Image.new("RGB", (2, 4), "red")
    generated = Image.new("RGB", (2, 8), "blue")

    restored = mk5.restore_prepared_source(generated, prepared, 2)

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


def test_soft_blend_vertical_seams_zero_preserves_hard_paste_result():
    mk5 = load_mk5_module_for_test()
    source = Image.new("RGB", (1, 4), "red")
    generated = Image.new("RGB", (1, 8), "blue")

    hard = generated.copy()
    hard.paste(source, (0, 2))
    blended = mk5.soft_blend_vertical_seams(generated, source, 2, 0)

    assert blended.tobytes() == hard.tobytes()


def test_soft_blend_vertical_seams_feathers_only_boundary_band():
    mk5 = load_mk5_module_for_test()
    source = Image.new("RGB", (1, 6), "red")
    generated = Image.new("RGB", (1, 10), "blue")

    blended = mk5.soft_blend_vertical_seams(generated, source, 2, 2)

    assert blended.getpixel((0, 0)) == (0, 0, 255)
    assert blended.getpixel((0, 9)) == (0, 0, 255)
    assert blended.getpixel((0, 2)) not in [(255, 0, 0), (0, 0, 255)]
    assert blended.getpixel((0, 3)) not in [(255, 0, 0), (0, 0, 255)]
    assert blended.getpixel((0, 4)) == (255, 0, 0)
    assert blended.getpixel((0, 5)) == (255, 0, 0)
    assert blended.getpixel((0, 6)) not in [(255, 0, 0), (0, 0, 255)]
    assert blended.getpixel((0, 7)) not in [(255, 0, 0), (0, 0, 255)]


def test_soft_blend_vertical_seams_partitions_oversized_two_sided_bands():
    mk5 = load_mk5_module_for_test()
    source = Image.new("RGB", (1, 5), "red")
    generated = Image.new("RGB", (1, 15), "blue")

    blended = mk5.soft_blend_vertical_seams(generated, source, 5, 99)

    assert [blended.getpixel((0, y)) for y in range(15)] == [
        (0, 0, 255),
        (0, 0, 255),
        (0, 0, 255),
        (0, 0, 255),
        (0, 0, 255),
        (63, 0, 191),
        (127, 0, 127),
        (191, 0, 63),
        (170, 0, 85),
        (85, 0, 170),
        (0, 0, 255),
        (0, 0, 255),
        (0, 0, 255),
        (0, 0, 255),
        (0, 0, 255),
    ]


@pytest.mark.parametrize(
    ("source_top", "expected"),
    [
        (
            0,
            [
                (255, 0, 0),
                (255, 0, 0),
                (191, 0, 63),
                (127, 0, 127),
                (63, 0, 191),
                (0, 0, 255),
                (0, 0, 255),
                (0, 0, 255),
            ],
        ),
        (
            3,
            [
                (0, 0, 255),
                (0, 0, 255),
                (0, 0, 255),
                (63, 0, 191),
                (127, 0, 127),
                (191, 0, 63),
                (255, 0, 0),
                (255, 0, 0),
            ],
        ),
    ],
)
def test_soft_blend_vertical_seams_preserves_one_sided_extent(source_top, expected):
    mk5 = load_mk5_module_for_test()
    source = Image.new("RGB", (1, 5), "red")
    generated = Image.new("RGB", (1, 8), "blue")

    blended = mk5.soft_blend_vertical_seams(generated, source, source_top, 3)

    assert [blended.getpixel((0, y)) for y in range(8)] == expected


@pytest.mark.parametrize(
    ("samples_save", "original_do_not_save_samples", "save_incomplete_images", "interrupted", "skipped", "expected"),
    [
        (True, False, False, False, False, True),
        (False, False, False, False, False, False),
        (True, True, False, False, False, False),
        (True, False, False, True, False, False),
        (True, False, False, False, True, False),
        (True, False, True, True, False, True),
    ],
)
def test_should_save_final_samples_matches_processing_policy(samples_save, original_do_not_save_samples, save_incomplete_images, interrupted, skipped, expected):
    mk5 = load_mk5_module_for_test()
    opts = types.SimpleNamespace(samples_save=samples_save, save_incomplete_images=save_incomplete_images)
    state = types.SimpleNamespace(interrupted=interrupted, skipped=skipped)

    assert mk5.should_save_final_samples(opts, state, original_do_not_save_samples) is expected


class FakeProcessed:
    def __init__(self, _p, images, seed, info, **kwargs):
        self.images = images
        self.seed = seed
        self.info = info
        self.width = _p.width
        self.height = _p.height
        self.all_seeds = kwargs.get("all_seeds")
        self.infotexts = kwargs.get("infotexts")


def make_p(source):
    return types.SimpleNamespace(
        prompt="base prompt",
        negative_prompt="bad prompt",
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
        inpainting_mask_invert=1,
        inpaint_full_res=True,
        extra_generation_params={},
        outpath_samples="samples",
        seed=42,
        subseed=84,
    )


def test_run_generates_one_final_image_restores_source_and_saves():
    mk5 = load_mk5_module_for_test()
    calls = []
    saved = []

    def fake_process(p):
        calls.append((p.init_images[0].size, p.image_mask.size, p.width, p.height, p.inpainting_mask_invert, p.do_not_save_grid))
        return FakeProcessed(
            p,
            [Image.new("RGB", (p.width, p.height), "blue"), Image.new("RGB", (p.width, p.height), "green")],
            101,
            "info-101",
        )

    mk5.Processed = FakeProcessed
    mk5.process_images = fake_process
    mk5.opts = types.SimpleNamespace(samples_save=True, save_incomplete_images=False, samples_format="png")
    mk5.images.save_image = lambda image, _path, _basename, seed, _prompt, _format, info, p: saved.append((image.size, seed, info, p.do_not_save_grid))
    mk5.state = types.SimpleNamespace(job="", job_count=0, interrupted=False, skipped=False)
    p = make_p(Image.new("RGB", (4, 4), "red"))

    result = mk5.Script().run(p, 8, 0.0, "Center", 1, 4, 0, 0, False, "")

    assert len(result.images) == 1
    assert result.seed == 101
    assert result.all_seeds == [101]
    assert result.infotexts == [result.info]
    assert "Size: 4x8" in result.info
    assert "Vertical Outpainting MK5 target height: 8" in result.info
    assert calls == [((4, 8), (4, 8), 4, 8, 0, True)]
    assert saved == [((4, 8), 101, result.info, True)]
    assert result.images[0].getpixel((0, 2)) == (255, 0, 0)


def test_run_generates_requested_number_of_final_variants():
    mk5 = load_mk5_module_for_test()
    calls = []
    saved = []

    def fake_process(p):
        seed = 100 + len(calls)
        calls.append((p.init_images[0].size, p.image_mask.size, p.width, p.height))
        return FakeProcessed(p, [Image.new("RGB", (p.width, p.height), (0, 0, seed))], seed, f"info-{seed}")

    mk5.Processed = FakeProcessed
    mk5.process_images = fake_process
    mk5.opts = types.SimpleNamespace(samples_save=True, save_incomplete_images=False, samples_format="png")
    mk5.images.save_image = lambda image, _path, _basename, seed, _prompt, _format, info, p: saved.append((image.size, seed, info, p.do_not_save_grid))
    mk5.state = types.SimpleNamespace(job="", job_count=0, interrupted=False, skipped=False)
    p = make_p(Image.new("RGB", (4, 4), "red"))

    result = mk5.Script().run(p, 8, 0.0, "Center", 1, 4, 0, 0, False, "", 3)

    assert len(result.images) == 3
    assert result.all_seeds == [100, 101, 102]
    assert result.infotexts == [result.info, result.info.replace("Seed: 100", "Seed: 101"), result.info.replace("Seed: 100", "Seed: 102")]
    assert p.extra_generation_params["Vertical Outpainting MK5 variants per input image"] == 3
    assert calls == [((4, 8), (4, 8), 4, 8)] * 3
    assert saved == [(image.size, seed, info, True) for image, seed, info in zip(result.images, result.all_seeds, result.infotexts)]


def test_run_records_soft_seam_blend_metadata_and_applies_blend():
    mk5 = load_mk5_module_for_test()
    mk5.Processed = FakeProcessed
    mk5.process_images = lambda p: FakeProcessed(p, [Image.new("RGB", (p.width, p.height), "blue")], 101, "info-101")
    mk5.opts = types.SimpleNamespace(samples_save=False, save_incomplete_images=False, samples_format="png")
    mk5.state = types.SimpleNamespace(job="", job_count=0, interrupted=False, skipped=False)
    p = make_p(Image.new("RGB", (1, 6), "red"))

    result = mk5.Script().run(p, 10, 0.0, "Center", 1, 4, 2, 0, False, "")

    assert result.images[0].getpixel((0, 2)) not in [(255, 0, 0), (0, 0, 255)]
    assert result.images[0].getpixel((0, 3)) not in [(255, 0, 0), (0, 0, 255)]
    assert result.images[0].getpixel((0, 4)) == (255, 0, 0)
    assert p.extra_generation_params["Vertical Outpainting MK5 soft seam blend"] == 2


def test_run_normalizes_mutated_generation_size_and_restores_prepared_source():
    mk5 = load_mk5_module_for_test()

    def fake_process(p):
        p.width = 6
        p.height = 10
        return FakeProcessed(p, [Image.new("RGB", (6, 10), "blue")], 101, "info-101")

    mk5.Processed = FakeProcessed
    mk5.process_images = fake_process
    mk5.opts = types.SimpleNamespace(samples_save=False, save_incomplete_images=False, samples_format="png")
    mk5.state = types.SimpleNamespace(job="", job_count=0, interrupted=False, skipped=False)
    source = Image.new("RGB", (4, 4), "red")

    result = mk5.Script().run(make_p(source), 8, 0.0, "Center", 1, 4, 0, 0, False, "")

    assert result.images[0].size == (4, 8)
    assert (result.width, result.height) == (4, 8)
    assert result.info != "info-101"
    assert result.infotexts == [result.info]
    assert "Size: 4x8" in result.info
    assert "Size: 6x10" not in result.info
    assert result.images[0].crop((0, 2, 4, 6)).tobytes() == source.tobytes()


def test_run_no_gap_returns_source_only_without_processing():
    mk5 = load_mk5_module_for_test()
    calls = []
    mk5.Processed = FakeProcessed
    mk5.process_images = lambda _p: calls.append("called")
    mk5.opts = types.SimpleNamespace(samples_save=False, save_incomplete_images=False, samples_format="png")
    mk5.state = types.SimpleNamespace(job="", job_count=0, interrupted=False, skipped=False)
    source = Image.new("RGB", (4, 4), "red")

    result = mk5.Script().run(make_p(source), 4, 0.0, "Center", 1, 4, 32, 0, False, "")

    assert calls == []
    assert len(result.images) == 1
    assert result.images[0].size == (4, 4)
    assert result.images[0].tobytes() == source.tobytes()


def test_run_scale_one_returns_prepared_source_only_without_processing():
    mk5 = load_mk5_module_for_test()
    calls = []
    mk5.Processed = FakeProcessed
    mk5.process_images = lambda _p: calls.append("called")
    mk5.opts = types.SimpleNamespace(samples_save=False, save_incomplete_images=False, samples_format="png")
    mk5.state = types.SimpleNamespace(job="", job_count=0, interrupted=False, skipped=False)
    source = Image.new("RGB", (4, 4))
    for y in range(source.height):
        for x in range(source.width):
            source.putpixel((x, y), (x * 40, y * 60, 0))

    result = mk5.Script().run(make_p(source), 8, 1.0, "Center", 1, 4, 32, 0, False, "")

    assert calls == []
    assert len(result.images) == 1
    assert result.images[0].size == (4, 8)
    assert result.images[0].tobytes() == mk5.prepare_scaled_source(source, 8, 1.0).tobytes()
    assert result.images[0].getbbox() == (0, 0, 4, 8)


def test_run_empty_generation_returns_no_placeholder_and_saves_nothing():
    mk5 = load_mk5_module_for_test()
    saved = []
    mk5.Processed = FakeProcessed
    mk5.process_images = lambda p: FakeProcessed(p, [], 101, "info-101")
    mk5.opts = types.SimpleNamespace(samples_save=True, save_incomplete_images=False, samples_format="png")
    mk5.images.save_image = lambda *args, **kwargs: saved.append(args)
    mk5.state = types.SimpleNamespace(job="", job_count=0, interrupted=False, skipped=False)

    result = mk5.Script().run(make_p(Image.new("RGB", (4, 4), "red")), 8, 0.0, "Center", 1, 4, 32, 0, False, "")

    assert result.images == []
    assert result.all_seeds == []
    assert result.infotexts == []
    assert saved == []


def test_run_restores_prompt_and_processing_state():
    mk5 = load_mk5_module_for_test()
    prompts_seen = []

    def fake_process(p):
        prompts_seen.append(p.prompt)
        p.inpaint_full_res = False
        return FakeProcessed(p, [Image.new("RGB", (p.width, p.height), "blue")], 101, "info-101")

    mk5.Processed = FakeProcessed
    mk5.process_images = fake_process
    mk5.opts = types.SimpleNamespace(samples_save=False, save_incomplete_images=False, samples_format="png")
    mk5.state = types.SimpleNamespace(job="", job_count=0, interrupted=False, skipped=False)
    p = make_p(Image.new("RGB", (4, 4), "red"))
    original_init_images = p.init_images

    mk5.Script().run(p, 8, 0.0, "Center", 1, 4, 32, 0, True, "continue naturally")

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
    assert p.inpainting_mask_invert == 1
    assert p.inpaint_full_res is True


class FakeAsymmetricBlurProcessing:
    def __init__(self, source):
        values = vars(make_p(source)).copy()
        values.pop("mask_blur")
        self.__dict__.update(values)
        self.mask_blur_x = 3
        self.mask_blur_y = 7

    @property
    def mask_blur(self):
        if self.mask_blur_x == self.mask_blur_y:
            return self.mask_blur_x
        return None

    @mask_blur.setter
    def mask_blur(self, value):
        if isinstance(value, int):
            self.mask_blur_x = value
            self.mask_blur_y = value


def test_run_restores_asymmetric_mask_blur_axes():
    mk5 = load_mk5_module_for_test()

    def fake_process(p):
        assert (p.mask_blur_x, p.mask_blur_y) == (4, 4)
        return FakeProcessed(p, [Image.new("RGB", (p.width, p.height), "blue")], 101, "info-101")

    mk5.Processed = FakeProcessed
    mk5.process_images = fake_process
    mk5.opts = types.SimpleNamespace(samples_save=False, save_incomplete_images=False, samples_format="png")
    mk5.state = types.SimpleNamespace(job="", job_count=0, interrupted=False, skipped=False)
    p = FakeAsymmetricBlurProcessing(Image.new("RGB", (4, 4), "red"))

    mk5.Script().run(p, 8, 0.0, "Center", 1, 4, 32, 0, False, "")

    assert (p.mask_blur_x, p.mask_blur_y) == (3, 7)
    assert p.mask_blur is None


def test_run_honors_save_suppression_and_records_metadata():
    mk5 = load_mk5_module_for_test()
    saved = []
    mk5.Processed = FakeProcessed
    mk5.process_images = lambda p: FakeProcessed(p, [Image.new("RGB", (p.width, p.height), "blue")], 101, "info-101")
    mk5.opts = types.SimpleNamespace(samples_save=True, save_incomplete_images=False, samples_format="png")
    mk5.images.save_image = lambda *args, **kwargs: saved.append(args)
    mk5.state = types.SimpleNamespace(job="", job_count=0, interrupted=True, skipped=False)
    p = make_p(Image.new("RGB", (4, 4), "red"))
    p.do_not_save_samples = True

    mk5.Script().run(p, 8, 0.25, "Bottom", 8, 4, 32, 0, True, "continue naturally")

    assert saved == []
    assert p.extra_generation_params["Vertical Outpainting MK5 target height"] == 8
    assert p.extra_generation_params["Vertical Outpainting MK5 scale"] == 0.25
    assert p.extra_generation_params["Vertical Outpainting MK5 placement"] == "Bottom"
    assert p.extra_generation_params["Vertical Outpainting MK5 seam size"] == 8
    assert p.extra_generation_params["Vertical Outpainting MK5 mask blur"] == 4
    assert p.extra_generation_params["Vertical Outpainting MK5 continue prompt injected"] is True


def test_run_suppresses_internal_init_save_and_restores_option():
    mk5 = load_mk5_module_for_test()
    saved = []

    def fake_process(p):
        assert mk5.opts.save_init_img is False
        return FakeProcessed(p, [Image.new("RGB", (p.width, p.height), "blue")], 101, "info-101")

    mk5.Processed = FakeProcessed
    mk5.process_images = fake_process
    mk5.opts = types.SimpleNamespace(
        samples_save=True,
        save_incomplete_images=False,
        samples_format="png",
        save_init_img=True,
    )
    mk5.images.save_image = lambda *args, **kwargs: saved.append(args)
    mk5.state = types.SimpleNamespace(job="", job_count=0, interrupted=False, skipped=False)
    p = make_p(Image.new("RGB", (4, 4), "red"))
    p.do_not_save_samples = True

    mk5.Script().run(p, 8, 0.0, "Center", 1, 4, 32, 0, False, "")

    assert mk5.opts.save_init_img is True
    assert saved == []


def test_run_returns_clear_errors_for_missing_source_and_small_target():
    mk5 = load_mk5_module_for_test()
    mk5.Processed = FakeProcessed

    missing = make_p(Image.new("RGB", (2, 4), "red"))
    missing.init_images = []
    missing_result = mk5.Script().run(missing, 8, 0.0, "Center", 1, 4, 32, 0, False, "")

    small = make_p(Image.new("RGB", (2, 4), "red"))
    small_result = mk5.Script().run(small, 2, 0.0, "Center", 1, 4, 32, 0, False, "")

    assert missing_result.info == "Vertical Outpainting Mk5 requires one source image."
    assert small_result.info == "Target height must be greater than or equal to the source image height."


def test_run_removes_temporary_mask_invert_when_original_object_lacked_it():
    mk5 = load_mk5_module_for_test()
    mk5.Processed = FakeProcessed
    mk5.process_images = lambda p: FakeProcessed(p, [Image.new("RGB", (p.width, p.height), "blue")], 101, "info-101")
    mk5.opts = types.SimpleNamespace(samples_save=False, save_incomplete_images=False, samples_format="png")
    mk5.state = types.SimpleNamespace(job="", job_count=0, interrupted=False, skipped=False)
    p = make_p(Image.new("RGB", (4, 4), "red"))
    delattr(p, "inpainting_mask_invert")

    mk5.Script().run(p, 8, 0.0, "Center", 1, 4, 32, 0, False, "")

    assert not hasattr(p, "inpainting_mask_invert")


def test_run_no_gap_saves_source_only_result_when_allowed():
    mk5 = load_mk5_module_for_test()
    saved = []
    mk5.Processed = FakeProcessed
    mk5.process_images = lambda _p: pytest.fail("process_images should not run for a blank mask")
    mk5.opts = types.SimpleNamespace(samples_save=True, save_incomplete_images=False, samples_format="png")
    mk5.images.save_image = lambda image, _path, _basename, seed, _prompt, _format, info, p: saved.append(
        (image.size, seed, info, p.width, p.height)
    )
    mk5.state = types.SimpleNamespace(job="", job_count=0, interrupted=False, skipped=False)

    result = mk5.Script().run(make_p(Image.new("RGB", (4, 4), "red")), 4, 1.0, "Center", 1, 4, 32, 0, False, "")

    assert len(result.images) == 1
    assert (result.width, result.height) == (4, 4)
    assert result.info
    assert result.infotexts == [result.info]
    assert "Size: 4x4" in result.info
    assert "Vertical Outpainting MK5 target height: 4" in result.info
    assert saved == [((4, 4), 42, result.info, 4, 4)]
