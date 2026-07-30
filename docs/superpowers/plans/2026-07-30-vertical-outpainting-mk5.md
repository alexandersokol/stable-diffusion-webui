# Vertical Outpainting Mk5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new img2img script named `Vertical Outpainting Mk5` that scales a source image into a taller target canvas, generates remaining vertical gaps, and restores the scaled source pixels exactly.

**Architecture:** Add one new script file, `scripts/vertical_outpainting_mk5.py`, with pure helpers for scale math, scaled/cropped source preparation, placement offsets, target canvas creation, mask construction, source restoration, and save policy. Add one focused test file, `test/test_vertical_outpainting_mk5.py`, that imports the script with lightweight stubs and verifies helpers plus `Script.run()` orchestration without loading Stable Diffusion models.

**Tech Stack:** Python 3.9, PIL/Pillow, Gradio script UI, AUTOMATIC1111 `modules.scripts.Script`, `modules.processing.Processed`, `modules.processing.process_images`, `modules.images`, `modules.shared.opts`, `modules.shared.state`, pytest.

## Global Constraints

- Create a new script; do not modify `scripts/vertical_outpainting_mk3.py`, `scripts/vertical_outpainting_mk4.py`, `scripts/poor_mans_outpainting.py`, or `scripts/outpainting_mk_2.py`.
- The script title must be `Vertical Outpainting Mk5`.
- The script must appear only in img2img.
- Only vertical target-canvas generation is in scope.
- Horizontal target resizing, placement variants, grids, zoom-out controls, ControlNet/IP-Adapter wiring, and soft source resynthesis are out of scope.
- The target width is always the source image width.
- `Target height` defaults to `2048`, ranges from `64` to `4096`, and uses step `64`.
- `Scale` ranges from `0.00` to `1.00`, step `0.05`, default `0.00`.
- `Placement` choices are `Center`, `Top`, and `Bottom`; default is `Center`.
- `Seam size` ranges from `0` to `128`, step `8`, default `32`.
- `Mask blur` ranges from `0` to `64`, step `1`, default `4`.
- The script returns only the completed final image.
- The script must not return, add, or save a grid.
- The script must not add new controls for denoising strength, sampler, CFG scale, batch size, or batch count.
- If `Scale = 1.00`, or if no mask pixels are white, return the prepared source-only target canvas without calling `process_images`.
- If generation returns no images for a non-empty mask, return an empty `Processed` result and do not save a placeholder.
- If prompt injection is enabled, restore the original prompt after processing.

---

## File Structure

- Create `scripts/vertical_outpainting_mk5.py`
  - Owns the Mk5 img2img script UI and generation flow.
  - Exposes pure helpers for tests:
    - `DEFAULT_CONTINUE_PROMPT: str`
    - `PLACEMENT_CHOICES: list[str]`
    - `MASKED_CONTENT_CHOICES: list[str]`
    - `compute_scale_factor(source_height: int, target_height: int, scale: float) -> float`
    - `prepare_scaled_source(source: Image.Image, target_height: int, scale: float) -> Image.Image`
    - `compute_vertical_offset(source_height: int, target_height: int, placement: str) -> int`
    - `create_target_canvas(prepared_source: Image.Image, target_height: int, source_top: int) -> Image.Image`
    - `build_outpaint_mask(source_size: tuple[int, int], target_height: int, source_top: int, seam_size: int) -> Image.Image`
    - `restore_prepared_source(generated: Image.Image, prepared_source: Image.Image, source_top: int) -> Image.Image`
    - `mask_has_white(mask: Image.Image) -> bool`
    - `should_save_final_samples(opts, state, original_do_not_save_samples: bool) -> bool`
    - `_append_continue_prompt(prompt: str, continue_prompt: str) -> str`
  - Keeps model-dependent processing inside `Script.run()`.
- Create `test/test_vertical_outpainting_mk5.py`
  - Loads `scripts/vertical_outpainting_mk5.py` with lightweight module stubs, matching the Mk4 tests.
  - Tests scale math, source preparation, masks, restoration behavior, save policy, and script orchestration.

---

### Task 1: Define Helper Contracts With Failing Tests

**Files:**
- Create: `test/test_vertical_outpainting_mk5.py`

**Interfaces:**
- Consumes future helpers from `scripts/vertical_outpainting_mk5.py`.
- Produces failing tests that define Mk5 geometry, masking, restoration, and save-policy behavior.

- [ ] **Step 1: Create the test module with import stubs**

Create `test/test_vertical_outpainting_mk5.py`:

```python
import importlib.util
import sys
import types
from pathlib import Path

import pytest
from PIL import Image


def load_mk5_module_for_test():
    import modules as modules_package

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "vertical_outpainting_mk5.py"
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
```

- [ ] **Step 2: Add scale-factor and prepared-source tests**

Append:

```python
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
```

- [ ] **Step 3: Add vertical-offset and target-canvas tests**

Append:

```python
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
```

- [ ] **Step 4: Add mask tests**

Append:

```python
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
```

- [ ] **Step 5: Add restoration and save-policy tests**

Append:

```python
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
```

- [ ] **Step 6: Run helper tests and verify they fail because the script does not exist yet**

Run: `D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_vertical_outpainting_mk5.py -q`

Expected: FAIL during import with `FileNotFoundError` for `scripts/vertical_outpainting_mk5.py`.

- [ ] **Step 7: Commit failing helper tests**

```bash
git add test/test_vertical_outpainting_mk5.py
git commit -m "test: define vertical outpainting mk5 helpers"
```

---

### Task 2: Implement Pure Mk5 Helpers

**Files:**
- Create: `scripts/vertical_outpainting_mk5.py`
- Modify: `test/test_vertical_outpainting_mk5.py`

**Interfaces:**
- Consumes helper tests from Task 1.
- Produces all pure helper functions named in the file structure.

- [ ] **Step 1: Create the script shell and constants**

Create `scripts/vertical_outpainting_mk5.py`:

```python
import gradio as gr
import modules.scripts as scripts
from PIL import Image, ImageDraw

from modules import images
from modules.processing import Processed, process_images
from modules.shared import opts, state


DEFAULT_CONTINUE_PROMPT = (
    "extend the image naturally, continue the existing scene, preserve composition, "
    "lighting, perspective, colors, style, and subject consistency"
)

PLACEMENT_CHOICES = ["Center", "Top", "Bottom"]
MASKED_CONTENT_CHOICES = ["fill", "original", "latent noise", "latent nothing"]
```

- [ ] **Step 2: Implement scale and source-preparation helpers**

Append:

```python
def compute_scale_factor(source_height, target_height, scale):
    source_height = int(source_height)
    target_height = int(target_height)
    scale = max(0.0, min(float(scale), 1.0))
    return 1.0 + scale * ((target_height / source_height) - 1.0)


def prepare_scaled_source(source, target_height, scale):
    source = source.convert("RGB")
    scale_factor = compute_scale_factor(source.height, target_height, scale)
    scaled_width = max(source.width, int(round(source.width * scale_factor)))
    scaled_height = min(int(target_height), max(1, int(round(source.height * scale_factor))))
    resized = source.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
    left = max(0, (scaled_width - source.width) // 2)
    return resized.crop((left, 0, left + source.width, scaled_height))
```

- [ ] **Step 3: Implement placement, canvas, prompt, and mask helpers**

Append:

```python
def compute_vertical_offset(source_height, target_height, placement):
    gap = int(target_height) - int(source_height)
    if gap < 0:
        raise ValueError("Target height must be greater than or equal to the source image height.")
    if placement == "Top":
        return 0
    if placement == "Bottom":
        return gap
    if placement == "Center":
        return gap // 2
    raise ValueError(f"Unknown placement: {placement}")


def create_target_canvas(prepared_source, target_height, source_top):
    canvas = Image.new("RGB", (prepared_source.width, int(target_height)), "black")
    canvas.paste(prepared_source.convert("RGB"), (0, int(source_top)))
    return canvas


def _append_continue_prompt(prompt, continue_prompt):
    continue_prompt = (continue_prompt or "").strip()
    if not continue_prompt:
        return prompt
    if not prompt:
        return continue_prompt
    return f"{prompt}, {continue_prompt}"


def _clamped_seam(source_height, seam_size):
    return max(0, min(int(seam_size), int(source_height)))


def build_outpaint_mask(source_size, target_height, source_top, seam_size):
    source_width, source_height = source_size
    target_height = int(target_height)
    source_top = int(source_top)
    source_bottom = source_top + source_height
    seam = _clamped_seam(source_height, seam_size)
    mask = Image.new("L", (source_width, target_height), "black")
    draw = ImageDraw.Draw(mask)

    if source_top > 0:
        draw.rectangle((0, 0, source_width, min(target_height, source_top + seam) - 1), fill="white")
    if source_bottom < target_height:
        draw.rectangle((0, max(0, source_bottom - seam), source_width, target_height - 1), fill="white")

    return mask
```

- [ ] **Step 4: Implement restoration, mask check, and save-policy helpers**

Append:

```python
def restore_prepared_source(generated, prepared_source, source_top):
    result = generated.copy()
    result.paste(prepared_source.convert("RGB"), (0, int(source_top)))
    return result


def mask_has_white(mask):
    return mask.getbbox() is not None


def should_save_final_samples(opts_obj, state_obj, original_do_not_save_samples):
    return (
        bool(getattr(opts_obj, "samples_save", False))
        and not bool(original_do_not_save_samples)
        and (
            bool(getattr(opts_obj, "save_incomplete_images", False))
            or not bool(getattr(state_obj, "interrupted", False)) and not bool(getattr(state_obj, "skipped", False))
        )
    )
```

- [ ] **Step 5: Add minimal Script shell**

Append:

```python
class Script(scripts.Script):
    def title(self):
        return "Vertical Outpainting Mk5"

    def show(self, is_img2img):
        return is_img2img
```

- [ ] **Step 6: Run helper tests**

Run: `D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_vertical_outpainting_mk5.py -q`

Expected: PASS for all helper tests.

- [ ] **Step 7: Commit helpers**

```bash
git add scripts/vertical_outpainting_mk5.py test/test_vertical_outpainting_mk5.py
git commit -m "feat: add vertical outpainting mk5 helpers"
```

---

### Task 3: Add Mk5 Orchestration Tests

**Files:**
- Modify: `test/test_vertical_outpainting_mk5.py`

**Interfaces:**
- Consumes:
  - `Script.run(p, target_height, scale, placement, seam_size, mask_blur, inpainting_fill, inject_continue_prompt, continue_prompt) -> Processed`
  - helpers from Task 2.
- Produces failing tests defining UI/run behavior.

- [ ] **Step 1: Add fake Processed and processing object helpers**

Append:

```python
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
        inpainting_mask_invert=1,
        inpaint_full_res=True,
        extra_generation_params={},
        outpath_samples="samples",
        seed=42,
    )
```

- [ ] **Step 2: Add generation orchestration test**

Append:

```python
def test_run_generates_one_final_image_restores_source_and_saves():
    mk5 = load_mk5_module_for_test()
    calls = []
    saved = []

    def fake_process(p):
        calls.append((p.init_images[0].size, p.image_mask.size, p.width, p.height, p.inpainting_mask_invert, p.do_not_save_grid))
        return FakeProcessed(p, [Image.new("RGB", (p.width, p.height), "blue")], 101, "info-101")

    mk5.Processed = FakeProcessed
    mk5.process_images = fake_process
    mk5.opts = types.SimpleNamespace(samples_save=True, save_incomplete_images=False, samples_format="png")
    mk5.images.save_image = lambda image, _path, _basename, seed, _prompt, _format, info, p: saved.append((image.size, seed, info, p.do_not_save_grid))
    mk5.state = types.SimpleNamespace(job="", job_count=0, interrupted=False, skipped=False)
    p = make_p(Image.new("RGB", (4, 4), "red"))

    result = mk5.Script().run(p, 8, 0.0, "Center", 1, 4, 0, False, "")

    assert len(result.images) == 1
    assert result.seed == 101
    assert result.all_seeds == [101]
    assert result.infotexts == ["info-101"]
    assert calls == [((4, 8), (4, 8), 4, 8, 0, True)]
    assert saved == [((4, 8), 101, "info-101", True)]
    assert result.images[0].getpixel((0, 2)) == (255, 0, 0)
```

- [ ] **Step 3: Add no-gap and empty-generation tests**

Append:

```python
def test_run_no_gap_returns_source_only_without_processing():
    mk5 = load_mk5_module_for_test()
    calls = []
    mk5.Processed = FakeProcessed
    mk5.process_images = lambda _p: calls.append("called")
    mk5.opts = types.SimpleNamespace(samples_save=False, save_incomplete_images=False, samples_format="png")
    mk5.state = types.SimpleNamespace(job="", job_count=0, interrupted=False, skipped=False)
    source = Image.new("RGB", (4, 4), "red")

    result = mk5.Script().run(make_p(source), 4, 0.0, "Center", 1, 4, 0, False, "")

    assert calls == []
    assert len(result.images) == 1
    assert result.images[0].size == (4, 4)
    assert result.images[0].tobytes() == source.tobytes()


def test_run_empty_generation_returns_no_placeholder_and_saves_nothing():
    mk5 = load_mk5_module_for_test()
    saved = []
    mk5.Processed = FakeProcessed
    mk5.process_images = lambda p: FakeProcessed(p, [], 101, "info-101")
    mk5.opts = types.SimpleNamespace(samples_save=True, save_incomplete_images=False, samples_format="png")
    mk5.images.save_image = lambda *args, **kwargs: saved.append(args)
    mk5.state = types.SimpleNamespace(job="", job_count=0, interrupted=False, skipped=False)

    result = mk5.Script().run(make_p(Image.new("RGB", (4, 4), "red")), 8, 0.0, "Center", 1, 4, 0, False, "")

    assert result.images == []
    assert result.all_seeds == []
    assert result.infotexts == []
    assert saved == []
```

- [ ] **Step 4: Add state restoration, save suppression, and metadata tests**

Append:

```python
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

    mk5.Script().run(p, 8, 0.0, "Center", 1, 4, 0, True, "continue naturally")

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

    mk5.Script().run(p, 8, 0.25, "Bottom", 8, 4, 0, True, "continue naturally")

    assert saved == []
    assert p.extra_generation_params["Vertical Outpainting MK5 target height"] == 8
    assert p.extra_generation_params["Vertical Outpainting MK5 scale"] == 0.25
    assert p.extra_generation_params["Vertical Outpainting MK5 placement"] == "Bottom"
    assert p.extra_generation_params["Vertical Outpainting MK5 seam size"] == 8
    assert p.extra_generation_params["Vertical Outpainting MK5 mask blur"] == 4
    assert p.extra_generation_params["Vertical Outpainting MK5 continue prompt injected"] is True
```

- [ ] **Step 5: Add validation test**

Append:

```python
def test_run_returns_clear_errors_for_missing_source_and_small_target():
    mk5 = load_mk5_module_for_test()
    mk5.Processed = FakeProcessed

    missing = make_p(Image.new("RGB", (2, 4), "red"))
    missing.init_images = []
    missing_result = mk5.Script().run(missing, 8, 0.0, "Center", 1, 4, 0, False, "")

    small = make_p(Image.new("RGB", (2, 4), "red"))
    small_result = mk5.Script().run(small, 2, 0.0, "Center", 1, 4, 0, False, "")

    assert missing_result.info == "Vertical Outpainting Mk5 requires one source image."
    assert small_result.info == "Target height must be greater than or equal to the source image height."
```

- [ ] **Step 6: Run tests and verify orchestration tests fail before run implementation**

Run: `D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_vertical_outpainting_mk5.py -q`

Expected: helper tests pass and new orchestration tests fail with `AttributeError: 'Script' object has no attribute 'run'`.

- [ ] **Step 7: Commit failing orchestration tests**

```bash
git add test/test_vertical_outpainting_mk5.py
git commit -m "test: define vertical outpainting mk5 orchestration"
```

---

### Task 4: Implement Mk5 UI And Processing Flow

**Files:**
- Modify: `scripts/vertical_outpainting_mk5.py`

**Interfaces:**
- Consumes all tests from Tasks 1 and 3.
- Produces:
  - `Script.ui(self, is_img2img) -> list`
  - `Script.run(self, p, target_height, scale, placement, seam_size, mask_blur, inpainting_fill, inject_continue_prompt, continue_prompt) -> Processed`

- [ ] **Step 1: Implement `Script.ui()`**

Append inside `class Script`:

```python
    def ui(self, is_img2img):
        if not is_img2img:
            return None

        target_height = gr.Slider(label="Target height", minimum=64, maximum=4096, step=64, value=2048, elem_id=self.elem_id("target_height"))
        scale = gr.Slider(label="Scale", minimum=0.0, maximum=1.0, step=0.05, value=0.0, elem_id=self.elem_id("scale"))
        placement = gr.Dropdown(label="Placement", choices=PLACEMENT_CHOICES, value="Center", elem_id=self.elem_id("placement"))
        seam_size = gr.Slider(label="Seam size", minimum=0, maximum=128, step=8, value=32, elem_id=self.elem_id("seam_size"))
        mask_blur = gr.Slider(label="Mask blur", minimum=0, maximum=64, step=1, value=4, elem_id=self.elem_id("mask_blur"))
        inpainting_fill = gr.Radio(label="Masked content", choices=MASKED_CONTENT_CHOICES, value="fill", type="index", elem_id=self.elem_id("inpainting_fill"))
        inject_continue_prompt = gr.Checkbox(label="Inject continue prompt", value=False, elem_id=self.elem_id("inject_continue_prompt"))
        continue_prompt = gr.Textbox(label="Continue prompt", value=DEFAULT_CONTINUE_PROMPT, lines=2, elem_id=self.elem_id("continue_prompt"))

        return [target_height, scale, placement, seam_size, mask_blur, inpainting_fill, inject_continue_prompt, continue_prompt]
```

- [ ] **Step 2: Implement validation, metadata, and state capture**

Append inside `class Script`:

```python
    def run(self, p, target_height, scale, placement, seam_size, mask_blur, inpainting_fill, inject_continue_prompt, continue_prompt):
        if not getattr(p, "init_images", None):
            return Processed(p, [], p.seed, "Vertical Outpainting Mk5 requires one source image.")

        source = p.init_images[0].convert("RGB")
        target_height = int(target_height)
        if target_height < source.height:
            return Processed(p, [], p.seed, "Target height must be greater than or equal to the source image height.")

        prepared_source = prepare_scaled_source(source, target_height, scale)
        source_top = compute_vertical_offset(prepared_source.height, target_height, placement)
        target_canvas = create_target_canvas(prepared_source, target_height, source_top)
        mask = build_outpaint_mask(prepared_source.size, target_height, source_top, seam_size)

        original_prompt = p.prompt
        original_init_images = p.init_images
        original_image_mask = getattr(p, "image_mask", None)
        original_width = p.width
        original_height = p.height
        original_n_iter = p.n_iter
        original_batch_size = p.batch_size
        original_do_not_save_grid = p.do_not_save_grid
        original_do_not_save_samples = p.do_not_save_samples
        original_mask_blur = p.mask_blur
        original_inpainting_fill = p.inpainting_fill
        has_inpainting_mask_invert = hasattr(p, "inpainting_mask_invert")
        original_inpainting_mask_invert = getattr(p, "inpainting_mask_invert", None)
        has_inpaint_full_res = hasattr(p, "inpaint_full_res")
        original_inpaint_full_res = getattr(p, "inpaint_full_res", None)

        p.extra_generation_params["Vertical Outpainting MK5 target height"] = target_height
        p.extra_generation_params["Vertical Outpainting MK5 scale"] = float(scale)
        p.extra_generation_params["Vertical Outpainting MK5 placement"] = placement
        p.extra_generation_params["Vertical Outpainting MK5 seam size"] = int(seam_size)
        p.extra_generation_params["Vertical Outpainting MK5 mask blur"] = int(mask_blur)
        p.extra_generation_params["Vertical Outpainting MK5 continue prompt injected"] = bool(inject_continue_prompt)
```

- [ ] **Step 3: Implement no-gap return and generation path**

Continue the method:

```python
        try:
            if not mask_has_white(mask):
                final_image = target_canvas
                if should_save_final_samples(opts, state, original_do_not_save_samples):
                    images.save_image(final_image, p.outpath_samples, "", p.seed, original_prompt, opts.samples_format, info="", p=p)
                return Processed(p, [final_image], p.seed, "", all_seeds=[p.seed], infotexts=[""])

            if inject_continue_prompt:
                p.prompt = _append_continue_prompt(p.prompt, continue_prompt)

            p.n_iter = 1
            p.batch_size = 1
            p.do_not_save_grid = True
            p.do_not_save_samples = True
            p.width = source.width
            p.height = target_height
            p.mask_blur = int(mask_blur)
            p.inpainting_fill = inpainting_fill
            p.inpainting_mask_invert = 0
            state.job_count = 1
            state.job = "Vertical Outpainting Mk5: final canvas"

            p.init_images = [target_canvas]
            p.image_mask = mask
            processed = process_images(p)

            if not processed.images:
                return Processed(p, [], p.seed, "Vertical Outpainting Mk5 generation returned no images.", all_seeds=[], infotexts=[])

            final_image = restore_prepared_source(processed.images[0], prepared_source, source_top)
            if should_save_final_samples(opts, state, original_do_not_save_samples):
                images.save_image(final_image, p.outpath_samples, "", processed.seed, original_prompt, opts.samples_format, info=processed.info, p=p)

            return Processed(p, [final_image], processed.seed, processed.info, all_seeds=[processed.seed], infotexts=[processed.info])
```

- [ ] **Step 4: Implement state restoration**

Finish the method:

```python
        finally:
            p.prompt = original_prompt
            p.init_images = original_init_images
            p.image_mask = original_image_mask
            p.width = original_width
            p.height = original_height
            p.n_iter = original_n_iter
            p.batch_size = original_batch_size
            p.do_not_save_grid = original_do_not_save_grid
            p.do_not_save_samples = original_do_not_save_samples
            p.mask_blur = original_mask_blur
            p.inpainting_fill = original_inpainting_fill
            if has_inpainting_mask_invert:
                p.inpainting_mask_invert = original_inpainting_mask_invert
            elif hasattr(p, "inpainting_mask_invert"):
                delattr(p, "inpainting_mask_invert")
            if has_inpaint_full_res:
                p.inpaint_full_res = original_inpaint_full_res
```

- [ ] **Step 5: Run Mk5 tests**

Run: `D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_vertical_outpainting_mk5.py -q`

Expected: PASS.

- [ ] **Step 6: Commit UI and processing flow**

```bash
git add scripts/vertical_outpainting_mk5.py test/test_vertical_outpainting_mk5.py
git commit -m "feat: add vertical outpainting mk5 script"
```

---

### Task 5: Regression Coverage And Verification

**Files:**
- Modify: `test/test_vertical_outpainting_mk5.py`
- Verify: `scripts/vertical_outpainting_mk5.py`

**Interfaces:**
- Consumes complete Mk5 implementation.
- Produces additional regression coverage and final verification evidence.

- [ ] **Step 1: Add a test for no inherited mask-invert attribute**

Append:

```python
def test_run_removes_temporary_mask_invert_when_original_object_lacked_it():
    mk5 = load_mk5_module_for_test()
    mk5.Processed = FakeProcessed
    mk5.process_images = lambda p: FakeProcessed(p, [Image.new("RGB", (p.width, p.height), "blue")], 101, "info-101")
    mk5.opts = types.SimpleNamespace(samples_save=False, save_incomplete_images=False, samples_format="png")
    mk5.state = types.SimpleNamespace(job="", job_count=0, interrupted=False, skipped=False)
    p = make_p(Image.new("RGB", (4, 4), "red"))
    delattr(p, "inpainting_mask_invert")

    mk5.Script().run(p, 8, 0.0, "Center", 1, 4, 0, False, "")

    assert not hasattr(p, "inpainting_mask_invert")
```

- [ ] **Step 2: Add a test for source-only save behavior**

Append:

```python
def test_run_no_gap_saves_source_only_result_when_allowed():
    mk5 = load_mk5_module_for_test()
    saved = []
    mk5.Processed = FakeProcessed
    mk5.process_images = lambda _p: pytest.fail("process_images should not run for a blank mask")
    mk5.opts = types.SimpleNamespace(samples_save=True, save_incomplete_images=False, samples_format="png")
    mk5.images.save_image = lambda image, _path, _basename, seed, _prompt, _format, info, p: saved.append((image.size, seed, info))
    mk5.state = types.SimpleNamespace(job="", job_count=0, interrupted=False, skipped=False)

    result = mk5.Script().run(make_p(Image.new("RGB", (4, 4), "red")), 4, 1.0, "Center", 1, 4, 0, False, "")

    assert len(result.images) == 1
    assert saved == [((4, 4), 42, "")]
```

- [ ] **Step 3: Run Mk5 tests**

Run: `D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_vertical_outpainting_mk5.py -q`

Expected: PASS.

- [ ] **Step 4: Run outpainting regression tests**

Run: `D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_vertical_outpainting_mk5.py test/test_vertical_outpainting_mk4.py test/test_vertical_outpainting_mk3.py test/test_outpainting_mk2.py -q`

Expected: PASS. Existing `PytestConfigWarning: Unknown config option: base_url` is acceptable.

- [ ] **Step 5: Compile new script and tests**

Run:

```bash
D:\stable-diffusion\envs\v1101\python.exe -m py_compile scripts/vertical_outpainting_mk5.py test/test_vertical_outpainting_mk5.py
```

Expected: no output and exit code `0`.

- [ ] **Step 6: Attempt ruff if available**

Run:

```bash
D:\stable-diffusion\envs\v1101\python.exe -m ruff check scripts/vertical_outpainting_mk5.py test/test_vertical_outpainting_mk5.py
```

Expected: pass if ruff is installed. If it fails with `No module named ruff`, record that ruff is unavailable and do not install dependencies.

- [ ] **Step 7: Attempt full test suite and record unrelated environment failures**

Run:

```bash
D:\stable-diffusion\envs\v1101\python.exe -m pytest -q
```

Expected: the full suite may fail during collection because this workspace already has unrelated extension/repository dependency issues such as missing `sd_dynamic_prompts`, `imwatermark`, and `fire`, plus existing unrelated import/setup issues. Record the exact result in the implementation summary.

- [ ] **Step 8: Commit regression coverage**

```bash
git add scripts/vertical_outpainting_mk5.py test/test_vertical_outpainting_mk5.py
git commit -m "test: cover vertical outpainting mk5 regressions"
```

If Steps 3-7 required no file edits beyond test additions, commit only the test file.
