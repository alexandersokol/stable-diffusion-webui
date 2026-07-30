# Vertical Outpainting Mk4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new img2img script named `Vertical Outpainting Mk4` that creates final-height vertical outpainting variants with selectable source preservation behavior.

**Architecture:** Add one new script file, `scripts/vertical_outpainting_mk4.py`, with pure helpers for placement offsets, target canvas creation, mask construction, and source restoration. Add one focused test file, `test/test_vertical_outpainting_mk4.py`, that imports the script with lightweight stubs and validates helper behavior plus `Script.run()` orchestration without loading Stable Diffusion models.

**Tech Stack:** Python 3.9, PIL/Pillow, Gradio script UI, AUTOMATIC1111 `modules.scripts.Script`, `modules.processing.Processed`, `modules.processing.process_images`, `modules.images`, `modules.shared.opts`, `modules.shared.state`, pytest.

## Global Constraints

- Create a new script; do not modify `scripts/vertical_outpainting_mk3.py`, `scripts/poor_mans_outpainting.py`, or `scripts/outpainting_mk_2.py`.
- The script title must be `Vertical Outpainting Mk4`.
- The script must appear only in img2img.
- Only vertical target-canvas generation is in scope.
- Horizontal resizing, grids, external reference-extension integrations, and automatic ControlNet/IP-Adapter wiring are out of scope.
- The target width is always the source image width.
- `Target height` defaults to `2048`, ranges from `64` to `4096`, and uses step `64`.
- `Source placement` choices are `All`, `Centered`, `Top`, `Bottom`, `Top biased`, and `Bottom biased`; default is `All`.
- `Source handling` choices are `Preserve source pixels`, `Blend seam only`, and `Soft resynthesis`; default is `Preserve source pixels`.
- `Seam size` ranges from `0` to `128`, step `8`, default `32`.
- `Mask blur` ranges from `0` to `64`, step `1`, default `4`.
- The script returns only completed final placement images.
- The script must not return, add, or save a grid.
- The script must not add new controls for denoising strength, sampler, CFG scale, batch size, or batch count.
- `Soft resynthesis` must force internal masked content to `original`.
- If prompt injection is enabled, restore the original prompt after processing.

---

## File Structure

- Create `scripts/vertical_outpainting_mk4.py`
  - Owns the Mk4 img2img script UI and generation flow.
  - Exposes pure helpers for tests:
    - `DEFAULT_CONTINUE_PROMPT: str`
    - `PLACEMENT_CHOICES: list[str]`
    - `SOURCE_HANDLING_CHOICES: list[str]`
    - `build_placement_offsets(source_height: int, target_height: int, placement: str) -> list[tuple[str, int]]`
    - `create_target_canvas(source: Image.Image, target_height: int, source_top: int) -> Image.Image`
    - `build_source_mask(source_size: tuple[int, int], target_height: int, source_top: int, source_handling: str, seam_size: int) -> Image.Image`
    - `restore_source_region(generated: Image.Image, source: Image.Image, source_top: int, source_handling: str, seam_size: int) -> Image.Image`
    - `_append_continue_prompt(prompt: str, continue_prompt: str) -> str`
  - Keeps model-dependent processing inside `Script.run()`.
- Create `test/test_vertical_outpainting_mk4.py`
  - Loads `scripts/vertical_outpainting_mk4.py` with lightweight module stubs, matching `test/test_vertical_outpainting_mk3.py`.
  - Tests placement offsets, masks, restoration behavior, and script orchestration.

---

### Task 1: Define Helper Contracts With Failing Tests

**Files:**
- Create: `test/test_vertical_outpainting_mk4.py`

**Interfaces:**
- Consumes: Future helpers from `scripts/vertical_outpainting_mk4.py`:
  - `build_placement_offsets(source_height: int, target_height: int, placement: str) -> list[tuple[str, int]]`
  - `create_target_canvas(source: PIL.Image.Image, target_height: int, source_top: int) -> PIL.Image.Image`
  - `build_source_mask(source_size: tuple[int, int], target_height: int, source_top: int, source_handling: str, seam_size: int) -> PIL.Image.Image`
  - `restore_source_region(generated: PIL.Image.Image, source: PIL.Image.Image, source_top: int, source_handling: str, seam_size: int) -> PIL.Image.Image`
- Produces: Failing tests that define Mk4 geometry, masking, and restoration behavior.

- [ ] **Step 1: Create the test module with script-loading stubs**

Use this structure:

```python
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
```

- [ ] **Step 2: Add placement offset tests**

Append:

```python
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
```

- [ ] **Step 3: Add target canvas tests**

Append:

```python
def test_create_target_canvas_keeps_source_width_and_pastes_at_offset():
    mk4 = load_mk4_module_for_test()
    source = Image.new("RGB", (4, 3), "red")

    canvas = mk4.create_target_canvas(source, 7, 2)

    assert canvas.size == (4, 7)
    assert canvas.getpixel((0, 0)) == (0, 0, 0)
    assert canvas.getpixel((1, 2)) == (255, 0, 0)
    assert canvas.getpixel((1, 4)) == (255, 0, 0)
    assert canvas.getpixel((1, 6)) == (0, 0, 0)
```

- [ ] **Step 4: Add mask tests for each source handling mode**

Append:

```python
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
```

- [ ] **Step 5: Add restoration tests**

Append:

```python
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
```

- [ ] **Step 6: Run helper tests and verify they fail because the script does not exist yet**

Run: `D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_vertical_outpainting_mk4.py -q`

Expected: FAIL during import with `FileNotFoundError` for `scripts/vertical_outpainting_mk4.py`.

- [ ] **Step 7: Commit the failing tests**

```bash
git add test/test_vertical_outpainting_mk4.py
git commit -m "test: define vertical outpainting mk4 helpers"
```

---

### Task 2: Implement Pure Mk4 Helpers

**Files:**
- Create: `scripts/vertical_outpainting_mk4.py`
- Modify: `test/test_vertical_outpainting_mk4.py`

**Interfaces:**
- Consumes: Helper tests from Task 1.
- Produces:
  - `build_placement_offsets(source_height: int, target_height: int, placement: str) -> list[tuple[str, int]]`
  - `create_target_canvas(source: Image.Image, target_height: int, source_top: int) -> Image.Image`
  - `build_source_mask(source_size: tuple[int, int], target_height: int, source_top: int, source_handling: str, seam_size: int) -> Image.Image`
  - `restore_source_region(generated: Image.Image, source: Image.Image, source_top: int, source_handling: str, seam_size: int) -> Image.Image`
  - `_append_continue_prompt(prompt: str, continue_prompt: str) -> str`

- [ ] **Step 1: Add the script shell, constants, and imports**

Create `scripts/vertical_outpainting_mk4.py` with:

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

PLACEMENT_CHOICES = ["All", "Centered", "Top", "Bottom", "Top biased", "Bottom biased"]
SOURCE_HANDLING_CHOICES = ["Preserve source pixels", "Blend seam only", "Soft resynthesis"]
MASKED_CONTENT_CHOICES = ["fill", "original", "latent noise", "latent nothing"]
SOFT_RESYNTHESIS_INPAINTING_FILL = MASKED_CONTENT_CHOICES.index("original")
```

- [ ] **Step 2: Implement placement offset helper**

Append:

```python
def _placement_offset(source_height, target_height, placement):
    gap = target_height - source_height
    if placement == "Centered":
        return gap // 2
    if placement == "Top":
        return 0
    if placement == "Bottom":
        return gap
    if placement == "Top biased":
        return gap // 4
    if placement == "Bottom biased":
        return (gap * 3) // 4
    raise ValueError(f"Unknown source placement: {placement}")


def build_placement_offsets(source_height, target_height, placement):
    source_height = int(source_height)
    target_height = int(target_height)
    if target_height < source_height:
        raise ValueError("Target height must be greater than or equal to the source image height.")

    names = PLACEMENT_CHOICES[1:] if placement == "All" else [placement]
    offsets = []
    seen_offsets = set()
    for name in names:
        offset = _placement_offset(source_height, target_height, name)
        if offset in seen_offsets:
            continue
        seen_offsets.add(offset)
        offsets.append((name, offset))
    return offsets
```

- [ ] **Step 3: Implement canvas and continue prompt helpers**

Append:

```python
def create_target_canvas(source, target_height, source_top):
    canvas = Image.new("RGB", (source.width, int(target_height)), "black")
    canvas.paste(source.convert("RGB"), (0, int(source_top)))
    return canvas


def _append_continue_prompt(prompt, continue_prompt):
    continue_prompt = (continue_prompt or "").strip()
    if not continue_prompt:
        return prompt
    if not prompt:
        return continue_prompt
    return f"{prompt}, {continue_prompt}"
```

- [ ] **Step 4: Implement mask helper**

Append:

```python
def _clamped_seam(source_height, seam_size):
    return max(0, min(int(seam_size), int(source_height)))


def build_source_mask(source_size, target_height, source_top, source_handling, seam_size):
    source_width, source_height = source_size
    target_height = int(target_height)
    source_top = int(source_top)
    source_bottom = source_top + source_height

    if source_handling == "Soft resynthesis":
        return Image.new("L", (source_width, target_height), "white")

    seam = _clamped_seam(source_height, seam_size)
    has_top_gap = source_top > 0
    has_bottom_gap = source_bottom < target_height
    mask = Image.new("L", (source_width, target_height), "black")
    draw = ImageDraw.Draw(mask)

    if has_top_gap:
        draw.rectangle((0, 0, source_width, min(target_height, source_top + seam) - 1), fill="white")
    if has_bottom_gap:
        draw.rectangle((0, max(0, source_bottom - seam), source_width, target_height - 1), fill="white")

    return mask
```

- [ ] **Step 5: Implement source restoration helper**

Append:

```python
def restore_source_region(generated, source, source_top, source_handling, seam_size):
    result = generated.copy()
    source = source.convert("RGB")
    source_top = int(source_top)

    if source_handling == "Soft resynthesis":
        return result

    if source_handling == "Preserve source pixels":
        result.paste(source, (0, source_top))
        return result

    if source_handling != "Blend seam only":
        raise ValueError(f"Unknown source handling: {source_handling}")

    target_height = result.height
    source_bottom = source_top + source.height
    seam = _clamped_seam(source.height, seam_size)
    crop_top = seam if source_top > 0 else 0
    crop_bottom = source.height - seam if source_bottom < target_height else source.height

    if crop_bottom > crop_top:
        protected = source.crop((0, crop_top, source.width, crop_bottom))
        result.paste(protected, (0, source_top + crop_top))

    return result
```

- [ ] **Step 6: Add a minimal `Script` class shell**

Append:

```python
class Script(scripts.Script):
    def title(self):
        return "Vertical Outpainting Mk4"

    def show(self, is_img2img):
        return is_img2img
```

- [ ] **Step 7: Run helper tests and verify they pass**

Run: `D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_vertical_outpainting_mk4.py -q`

Expected: PASS for all helper tests.

- [ ] **Step 8: Commit helpers**

```bash
git add scripts/vertical_outpainting_mk4.py test/test_vertical_outpainting_mk4.py
git commit -m "feat: add vertical outpainting mk4 helpers"
```

---

### Task 3: Add Script UI And Orchestration Tests

**Files:**
- Modify: `test/test_vertical_outpainting_mk4.py`

**Interfaces:**
- Consumes:
  - `Script.run(p, target_height, source_placement, source_handling, seam_size, mask_blur, inpainting_fill, inject_continue_prompt, continue_prompt)`
  - Helpers from Task 2.
- Produces: Failing tests defining `Script.run()` processing behavior.

- [ ] **Step 1: Add fake processed class and p factory helpers**

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
        extra_generation_params={},
        outpath_samples="samples",
        seed=42,
    )
```

- [ ] **Step 2: Add test that `run()` creates final placement outputs only and saves samples**

Append:

```python
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

    result = mk4.Script().run(p, 8, "All", "Preserve source pixels", 1, 4, 0, False, "")

    assert len(result.images) == 5
    assert result.all_seeds == [101, 102, 103, 104, 105]
    assert result.infotexts == ["info-101", "info-102", "info-103", "info-104", "info-105"]
    assert calls == [((4, 8), (4, 8), 4, 8, True)] * 5
    assert saved == [((4, 8), seed, f"info-{seed}", True) for seed in result.all_seeds]
```

- [ ] **Step 3: Add test that source handling changes post-processing restoration**

Append:

```python
def test_run_preserves_source_pixels_or_leaves_soft_resynthesis_generated():
    mk4 = load_mk4_module_for_test()

    def fake_process(p):
        return FakeProcessed(p, [Image.new("RGB", (p.width, p.height), "blue")], 101, "info-101")

    mk4.Processed = FakeProcessed
    mk4.process_images = fake_process
    mk4.opts = types.SimpleNamespace(samples_save=False, samples_format="png")
    mk4.state = types.SimpleNamespace(job="", job_count=0)
    source = Image.new("RGB", (2, 4), "red")

    preserve = mk4.Script().run(make_p(source), 8, "Centered", "Preserve source pixels", 1, 4, 0, False, "")
    soft = mk4.Script().run(make_p(source), 8, "Centered", "Soft resynthesis", 1, 4, 0, False, "")

    assert preserve.images[0].getpixel((0, 2)) == (255, 0, 0)
    assert preserve.images[0].getpixel((0, 3)) == (255, 0, 0)
    assert soft.images[0].getpixel((0, 2)) == (0, 0, 255)
```

- [ ] **Step 4: Add test for soft resynthesis forcing masked content to original**

Append:

```python
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

    mk4.Script().run(make_p(Image.new("RGB", (2, 4), "red")), 8, "Centered", "Soft resynthesis", 1, 4, 3, False, "")

    assert inpainting_fill_values == [mk4.SOFT_RESYNTHESIS_INPAINTING_FILL]
```

- [ ] **Step 5: Add test for prompt and p-state restoration**

Append:

```python
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

    mk4.Script().run(p, 8, "Centered", "Preserve source pixels", 1, 4, 0, True, "continue naturally")

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
```

- [ ] **Step 6: Add tests for empty input and invalid target height**

Append:

```python
def test_run_returns_clear_errors_for_missing_source_and_small_target():
    mk4 = load_mk4_module_for_test()
    mk4.Processed = FakeProcessed

    missing = make_p(Image.new("RGB", (2, 4), "red"))
    missing.init_images = []
    missing_result = mk4.Script().run(missing, 8, "Centered", "Preserve source pixels", 1, 4, 0, False, "")

    small = make_p(Image.new("RGB", (2, 4), "red"))
    small_result = mk4.Script().run(small, 2, "Centered", "Preserve source pixels", 1, 4, 0, False, "")

    assert missing_result.info == "Vertical Outpainting Mk4 requires one source image."
    assert small_result.info == "Target height must be greater than or equal to the source image height."
```

- [ ] **Step 7: Run orchestration tests and verify they fail because `Script.run()` is missing**

Run: `D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_vertical_outpainting_mk4.py -q`

Expected: FAIL with `AttributeError: 'Script' object has no attribute 'run'` for the new orchestration tests.

- [ ] **Step 8: Commit failing orchestration tests**

```bash
git add test/test_vertical_outpainting_mk4.py
git commit -m "test: define vertical outpainting mk4 orchestration"
```

---

### Task 4: Implement Mk4 UI And Processing Flow

**Files:**
- Modify: `scripts/vertical_outpainting_mk4.py`

**Interfaces:**
- Consumes:
  - Helper functions from Task 2.
  - Orchestration tests from Task 3.
- Produces:
  - `Script.ui(self, is_img2img) -> list`
  - `Script.run(self, p, target_height, source_placement, source_handling, seam_size, mask_blur, inpainting_fill, inject_continue_prompt, continue_prompt) -> Processed`

- [ ] **Step 1: Implement `Script.ui()`**

Append this method inside `class Script`:

```python
    def ui(self, is_img2img):
        if not is_img2img:
            return None

        target_height = gr.Slider(label="Target height", minimum=64, maximum=4096, step=64, value=2048, elem_id=self.elem_id("target_height"))
        source_placement = gr.Dropdown(label="Source placement", choices=PLACEMENT_CHOICES, value="All", elem_id=self.elem_id("source_placement"))
        source_handling = gr.Dropdown(label="Source handling", choices=SOURCE_HANDLING_CHOICES, value="Preserve source pixels", elem_id=self.elem_id("source_handling"))
        seam_size = gr.Slider(label="Seam size", minimum=0, maximum=128, step=8, value=32, elem_id=self.elem_id("seam_size"))
        mask_blur = gr.Slider(label="Mask blur", minimum=0, maximum=64, step=1, value=4, elem_id=self.elem_id("mask_blur"))
        inpainting_fill = gr.Radio(label="Masked content", choices=MASKED_CONTENT_CHOICES, value="fill", type="index", elem_id=self.elem_id("inpainting_fill"))
        inject_continue_prompt = gr.Checkbox(label="Inject continue prompt", value=False, elem_id=self.elem_id("inject_continue_prompt"))
        continue_prompt = gr.Textbox(label="Continue prompt", value=DEFAULT_CONTINUE_PROMPT, lines=2, elem_id=self.elem_id("continue_prompt"))

        return [target_height, source_placement, source_handling, seam_size, mask_blur, inpainting_fill, inject_continue_prompt, continue_prompt]
```

- [ ] **Step 2: Implement `Script.run()` state capture and validation**

Append this method inside `class Script`:

```python
    def run(self, p, target_height, source_placement, source_handling, seam_size, mask_blur, inpainting_fill, inject_continue_prompt, continue_prompt):
        if not getattr(p, "init_images", None):
            return Processed(p, [], p.seed, "Vertical Outpainting Mk4 requires one source image.")

        source = p.init_images[0].convert("RGB")
        target_height = int(target_height)
        try:
            placements = build_placement_offsets(source.height, target_height, source_placement)
        except ValueError as exc:
            return Processed(p, [], p.seed, str(exc))

        if not placements:
            return Processed(p, [], p.seed, "No valid source placement selected.")

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
```

- [ ] **Step 3: Implement `Script.run()` processing loop**

Continue the method with:

```python
        p.extra_generation_params["Vertical Outpainting MK4 target height"] = target_height
        p.extra_generation_params["Vertical Outpainting MK4 source placement"] = source_placement
        p.extra_generation_params["Vertical Outpainting MK4 source handling"] = source_handling
        p.extra_generation_params["Vertical Outpainting MK4 seam size"] = int(seam_size)
        p.extra_generation_params["Vertical Outpainting MK4 mask blur"] = int(mask_blur)
        p.extra_generation_params["Vertical Outpainting MK4 continue prompt injected"] = bool(inject_continue_prompt)

        final_images = []
        variant_seeds = []
        variant_infos = []

        try:
            if inject_continue_prompt:
                p.prompt = _append_continue_prompt(p.prompt, continue_prompt)

            p.n_iter = 1
            p.batch_size = 1
            p.do_not_save_grid = True
            p.do_not_save_samples = True
            p.width = source.width
            p.height = target_height
            p.mask_blur = int(mask_blur)
            p.inpainting_fill = SOFT_RESYNTHESIS_INPAINTING_FILL if source_handling == "Soft resynthesis" else inpainting_fill
            state.job_count = len(placements)

            for index, (placement_name, source_top) in enumerate(placements):
                state.job = f"{placement_name}: final canvas {index + 1} out of {len(placements)}"
                target_canvas = create_target_canvas(source, target_height, source_top)
                mask = build_source_mask(source.size, target_height, source_top, source_handling, seam_size)

                p.init_images = [target_canvas]
                p.image_mask = mask
                processed = process_images(p)

                generated = processed.images[0] if processed.images else target_canvas
                final_image = restore_source_region(generated, source, source_top, source_handling, seam_size)
                final_images.append(final_image)
                variant_seeds.append(processed.seed)
                variant_infos.append(processed.info)
                p.seed = processed.seed + 1
```

- [ ] **Step 4: Implement saving, return, and state restoration**

Finish the method with:

```python
            if opts.samples_save:
                for image, seed, info in zip(final_images, variant_seeds, variant_infos):
                    images.save_image(image, p.outpath_samples, "", seed, original_prompt, opts.samples_format, info=info, p=p)

            return Processed(p, final_images, variant_seeds[0], variant_infos[0], all_seeds=variant_seeds, infotexts=variant_infos)
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
```

- [ ] **Step 5: Run Mk4 tests and verify they pass**

Run: `D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_vertical_outpainting_mk4.py -q`

Expected: PASS.

- [ ] **Step 6: Commit UI and processing flow**

```bash
git add scripts/vertical_outpainting_mk4.py test/test_vertical_outpainting_mk4.py
git commit -m "feat: add vertical outpainting mk4 script"
```

---

### Task 5: Add Metadata And Regression Coverage

**Files:**
- Modify: `test/test_vertical_outpainting_mk4.py`
- Modify: `scripts/vertical_outpainting_mk4.py`

**Interfaces:**
- Consumes: Complete `Script.run()` from Task 4.
- Produces: Regression coverage for metadata, explicit placement, and source-edge seam behavior.

- [ ] **Step 1: Add metadata and explicit placement test**

Append:

```python
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

    result = mk4.Script().run(p, 8, "Bottom", "Blend seam only", 1, 4, 0, True, "continue naturally")

    assert len(result.images) == 1
    assert mask_values_by_row(masks[0]) == [255, 255, 255, 255, 255, 0, 0, 0]
    assert p.extra_generation_params["Vertical Outpainting MK4 target height"] == 8
    assert p.extra_generation_params["Vertical Outpainting MK4 source placement"] == "Bottom"
    assert p.extra_generation_params["Vertical Outpainting MK4 source handling"] == "Blend seam only"
    assert p.extra_generation_params["Vertical Outpainting MK4 seam size"] == 1
    assert p.extra_generation_params["Vertical Outpainting MK4 mask blur"] == 4
    assert p.extra_generation_params["Vertical Outpainting MK4 continue prompt injected"] is True
```

- [ ] **Step 2: Add seam clamp test**

Append:

```python
def test_seam_size_is_clamped_to_source_height():
    mk4 = load_mk4_module_for_test()

    mask = mk4.build_source_mask((2, 4), 8, 2, "Blend seam only", 999)
    restored = mk4.restore_source_region(Image.new("RGB", (2, 8), "blue"), Image.new("RGB", (2, 4), "red"), 2, "Blend seam only", 999)

    assert mask_values_by_row(mask) == [255] * 8
    assert restored.tobytes() == Image.new("RGB", (2, 8), "blue").tobytes()
```

- [ ] **Step 3: Run Mk4 tests and fix any mismatch by adjusting implementation, not the spec**

Run: `D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_vertical_outpainting_mk4.py -q`

Expected: PASS.

- [ ] **Step 4: Run Mk3 regression tests with Mk4 tests**

Run: `D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_vertical_outpainting_mk4.py test/test_vertical_outpainting_mk3.py test/test_outpainting_mk2.py -q`

Expected: PASS. Existing warnings are acceptable if all selected tests pass.

- [ ] **Step 5: Commit regression coverage**

```bash
git add scripts/vertical_outpainting_mk4.py test/test_vertical_outpainting_mk4.py
git commit -m "test: cover vertical outpainting mk4 metadata"
```

---

### Task 6: Final Verification

**Files:**
- Verify: `scripts/vertical_outpainting_mk4.py`
- Verify: `test/test_vertical_outpainting_mk4.py`

**Interfaces:**
- Consumes: Completed Mk4 script and tests.
- Produces: Evidence that the new script imports, compiles, and passes focused regression tests.

- [ ] **Step 1: Compile the new script and tests**

Run:

```bash
D:\stable-diffusion\envs\v1101\python.exe -m py_compile scripts/vertical_outpainting_mk4.py test/test_vertical_outpainting_mk4.py
```

Expected: no output and exit code `0`.

- [ ] **Step 2: Run targeted Mk4 test suite**

Run:

```bash
D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_vertical_outpainting_mk4.py -q
```

Expected: all Mk4 tests pass.

- [ ] **Step 3: Run outpainting regression suite**

Run:

```bash
D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_vertical_outpainting_mk4.py test/test_vertical_outpainting_mk3.py test/test_outpainting_mk2.py -q
```

Expected: all selected tests pass. Existing `PytestConfigWarning: Unknown config option: base_url` is acceptable.

- [ ] **Step 4: Attempt ruff if available**

Run:

```bash
D:\stable-diffusion\envs\v1101\python.exe -m ruff check scripts/vertical_outpainting_mk4.py test/test_vertical_outpainting_mk4.py
```

Expected: pass if ruff is installed. If it fails with `No module named ruff`, record that ruff is unavailable and do not install dependencies.

- [ ] **Step 5: Attempt full test suite and record unrelated environment failures**

Run:

```bash
D:\stable-diffusion\envs\v1101\python.exe -m pytest -q
```

Expected: the full suite may fail during collection because this workspace already had unrelated missing optional dependencies such as `sd_dynamic_prompts`, `imwatermark`, and `fire`. Record the exact result in the final implementation summary.

- [ ] **Step 6: Commit any final fixes**

If final verification required edits:

```bash
git add scripts/vertical_outpainting_mk4.py test/test_vertical_outpainting_mk4.py
git commit -m "fix: finalize vertical outpainting mk4"
```

If no files changed, do not create an empty commit.
