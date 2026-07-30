# Vertical Outpainting Mk3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new img2img script named `Vertical Outpainting Mk3` that produces completed vertical outpainting variants from sequential preset passes.

**Architecture:** Add one new script file, `scripts/vertical_outpainting_mk3.py`, copied from the useful processing shape of `scripts/poor_mans_outpainting.py` but refactored into helpers for preset selection, centered zoom, one vertical expansion pass, and script orchestration. Add one focused test file, `test/test_vertical_outpainting_mk3.py`, that imports the script with stubs and validates the pure helper behavior without loading Stable Diffusion models.

**Tech Stack:** Python 3.9, PIL/Pillow, Gradio script UI, AUTOMATIC1111 `modules.scripts.Script`, `modules.processing.Processed`, `modules.processing.process_images`, `modules.images`, `modules.shared.opts`, `modules.shared.state`, pytest.

## Global Constraints

- Create a new script; do not modify `scripts/poor_mans_outpainting.py` or `scripts/outpainting_mk_2.py`.
- The script title must be `Vertical Outpainting Mk3`.
- The script must appear only in img2img.
- Only vertical directions are in scope: `up` and `down`.
- `Pixels to expand` must range from `8` to `384`, step `8`, default `384`.
- The script returns only completed final preset images, never intermediate staged images.
- The script must not return, add, or save a grid.
- Each valid preset must finish at total vertical growth `2 * Pixels to expand`.
- Each preset must start from the same prepared image.
- `Zoom in` scales image content around the center and crops back to the original canvas size before outpainting.
- If prompt injection is enabled, restore the original prompt after processing.

---

## File Structure

- Create `scripts/vertical_outpainting_mk3.py`
  - Owns the MK3 img2img script UI and generation flow.
  - Exposes pure helpers for tests:
    - `DEFAULT_CONTINUE_PROMPT: str`
    - `SHIFT_PRESET_CHOICES: list[str]`
    - `build_shift_sequences(direction: list[str], shift_preset: str, pixels: int) -> list[tuple[str, list[tuple[str, int]]]]`
    - `zoom_content_centered(image: Image.Image, zoom: float) -> Image.Image`
  - Keeps model-dependent processing inside `Script.run()` and `_expand_vertical_once(...)`.
- Create `test/test_vertical_outpainting_mk3.py`
  - Loads `scripts/vertical_outpainting_mk3.py` with lightweight module stubs, matching `test/test_outpainting_mk2.py`.
  - Tests preset filtering and centered zoom behavior.

---

### Task 1: Add Helper Tests

**Files:**
- Create: `test/test_vertical_outpainting_mk3.py`

**Interfaces:**
- Consumes: Future helpers from `scripts/vertical_outpainting_mk3.py`:
  - `build_shift_sequences(direction: list[str], shift_preset: str, pixels: int) -> list[tuple[str, list[tuple[str, int]]]]`
  - `zoom_content_centered(image: PIL.Image.Image, zoom: float) -> PIL.Image.Image`
- Produces: Failing tests that define the helper contract.

- [ ] **Step 1: Create the test module with script-loading stubs**

Use this structure:

```python
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
```

- [ ] **Step 2: Add tests for all-direction preset expansion**

Append:

```python
def test_build_shift_sequences_all_with_both_directions():
    mk3 = load_mk3_module_for_test()

    assert mk3.build_shift_sequences(["up", "down"], "All", 384) == [
        ("Centered", [("up", 384), ("down", 384)]),
        ("Top", [("up", 384), ("up", 384)]),
        ("Bottom", [("down", 384), ("down", 384)]),
        ("Top Centered", [("up", 384), ("down", 192), ("up", 192)]),
        ("Bottom Centered", [("down", 384), ("up", 192), ("down", 192)]),
    ]
```

- [ ] **Step 3: Add tests for single-direction filtering**

Append:

```python
def test_build_shift_sequences_filters_single_direction_presets():
    mk3 = load_mk3_module_for_test()

    assert mk3.build_shift_sequences(["up"], "All", 256) == [
        ("Top", [("up", 256), ("up", 256)]),
    ]
    assert mk3.build_shift_sequences(["down"], "All", 256) == [
        ("Bottom", [("down", 256), ("down", 256)]),
    ]
    assert mk3.build_shift_sequences(["up"], "Centered", 256) == []
```

- [ ] **Step 4: Add tests for explicit preset selection and no direction**

Append:

```python
def test_build_shift_sequences_respects_explicit_preset():
    mk3 = load_mk3_module_for_test()

    assert mk3.build_shift_sequences(["up", "down"], "Top Centered", 128) == [
        ("Top Centered", [("up", 128), ("down", 64), ("up", 64)]),
    ]
    assert mk3.build_shift_sequences([], "All", 128) == []
```

- [ ] **Step 5: Add tests for centered zoom**

Append:

```python
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
```

- [ ] **Step 6: Run tests to verify they fail**

Run:

```bash
pytest test/test_vertical_outpainting_mk3.py -q
```

Expected: FAIL because `scripts/vertical_outpainting_mk3.py` does not exist.

- [ ] **Step 7: Commit the failing tests**

```bash
git add test/test_vertical_outpainting_mk3.py
git commit -m "test: define vertical outpainting mk3 helpers"
```

---

### Task 2: Implement Pure MK3 Helpers And UI Shell

**Files:**
- Create: `scripts/vertical_outpainting_mk3.py`
- Modify: `test/test_vertical_outpainting_mk3.py` only if test import stubs need a small correction.

**Interfaces:**
- Consumes: Tests from Task 1.
- Produces:
  - `DEFAULT_CONTINUE_PROMPT: str`
  - `SHIFT_PRESET_CHOICES: list[str]`
  - `build_shift_sequences(direction, shift_preset, pixels)`
  - `zoom_content_centered(image, zoom)`
  - `class Script(scripts.Script)` with `title()`, `show()`, and `ui()`.

- [ ] **Step 1: Add imports and constants**

Create `scripts/vertical_outpainting_mk3.py` with:

```python
import math

import gradio as gr
import modules.scripts as scripts
from PIL import Image, ImageDraw

from modules import devices, images
from modules.processing import Processed, process_images
from modules.shared import opts, state


DEFAULT_CONTINUE_PROMPT = (
    "extend the image naturally, continue the existing scene, preserve composition, "
    "lighting, perspective, colors, style, and subject consistency"
)

SHIFT_PRESET_CHOICES = [
    "All",
    "Centered",
    "Top",
    "Bottom",
    "Top Centered",
    "Bottom Centered",
]
```

- [ ] **Step 2: Implement preset helpers**

Add:

```python
def _preset_sequences(pixels):
    half_pixels = pixels // 2
    return {
        "Centered": [("up", pixels), ("down", pixels)],
        "Top": [("up", pixels), ("up", pixels)],
        "Bottom": [("down", pixels), ("down", pixels)],
        "Top Centered": [("up", pixels), ("down", half_pixels), ("up", half_pixels)],
        "Bottom Centered": [("down", pixels), ("up", half_pixels), ("down", half_pixels)],
    }


def _is_sequence_allowed(sequence, selected_directions):
    return all(direction in selected_directions for direction, _ in sequence)


def build_shift_sequences(direction, shift_preset, pixels):
    selected_directions = set(direction or [])
    sequences = _preset_sequences(int(pixels))

    if shift_preset == "All":
        names = ["Centered", "Top", "Bottom", "Top Centered", "Bottom Centered"]
    else:
        names = [shift_preset]

    return [
        (name, sequences[name])
        for name in names
        if name in sequences and _is_sequence_allowed(sequences[name], selected_directions)
    ]
```

- [ ] **Step 3: Implement centered zoom helper**

Add:

```python
def zoom_content_centered(image, zoom):
    zoom = float(zoom)
    if zoom <= 1.0:
        return image.copy()

    width, height = image.size
    resized_width = max(width, int(round(width * zoom)))
    resized_height = max(height, int(round(height * zoom)))
    resized = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

    left = (resized_width - width) // 2
    top = (resized_height - height) // 2
    return resized.crop((left, top, left + width, top + height))
```

- [ ] **Step 4: Add `Script` title/show/ui methods**

Add:

```python
class Script(scripts.Script):
    def title(self):
        return "Vertical Outpainting Mk3"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        if not is_img2img:
            return None

        pixels = gr.Slider(label="Pixels to expand", minimum=8, maximum=384, step=8, value=384, elem_id=self.elem_id("pixels"))
        mask_blur = gr.Slider(label="Mask blur", minimum=0, maximum=64, step=1, value=4, elem_id=self.elem_id("mask_blur"))
        inpainting_fill = gr.Radio(label="Masked content", choices=["fill", "original", "latent noise", "latent nothing"], value="fill", type="index", elem_id=self.elem_id("inpainting_fill"))
        direction = gr.CheckboxGroup(label="Outpainting direction", choices=["up", "down"], value=["up", "down"], elem_id=self.elem_id("direction"))
        shift_preset = gr.Dropdown(label="Shift preset", choices=SHIFT_PRESET_CHOICES, value="All", elem_id=self.elem_id("shift_preset"))
        zoom = gr.Slider(label="Zoom in", minimum=1.0, maximum=3.0, step=0.05, value=1.0, elem_id=self.elem_id("zoom"))
        inject_continue_prompt = gr.Checkbox(label="Inject continue prompt", value=False, elem_id=self.elem_id("inject_continue_prompt"))
        continue_prompt = gr.Textbox(label="Continue prompt", value=DEFAULT_CONTINUE_PROMPT, lines=2, elem_id=self.elem_id("continue_prompt"))

        return [pixels, mask_blur, inpainting_fill, direction, shift_preset, zoom, inject_continue_prompt, continue_prompt]
```

- [ ] **Step 5: Run helper tests**

Run:

```bash
pytest test/test_vertical_outpainting_mk3.py -q
```

Expected: PASS.

- [ ] **Step 6: Compile the new script**

Run:

```bash
python -m py_compile scripts/vertical_outpainting_mk3.py
```

Expected: PASS with no output.

- [ ] **Step 7: Commit helpers and UI shell**

```bash
git add scripts/vertical_outpainting_mk3.py test/test_vertical_outpainting_mk3.py
git commit -m "feat: add vertical outpainting mk3 shell"
```

---

### Task 3: Implement One-Pass Vertical Outpainting

**Files:**
- Modify: `scripts/vertical_outpainting_mk3.py`

**Interfaces:**
- Consumes:
  - `zoom_content_centered(image, zoom)`
  - `build_shift_sequences(direction, shift_preset, pixels)`
- Produces:
  - `_expand_vertical_once(p, init_img, expand_pixels, pass_direction, mask_blur) -> tuple[Image.Image, int | None, str | None]`
  - The helper returns one completed image for one vertical pass plus the first seed/info from its internal tile processing.

- [ ] **Step 1: Add `_round_up_to_64`**

Add near the helpers:

```python
def _round_up_to_64(value):
    return math.ceil(value / 64) * 64
```

- [ ] **Step 2: Add `_expand_vertical_once` by adapting Poor man's outpainting**

Add:

```python
def _expand_vertical_once(p, init_img, expand_pixels, pass_direction, mask_blur):
    initial_seed = None
    initial_info = None
    up = int(expand_pixels) if pass_direction == "up" else 0
    down = int(expand_pixels) if pass_direction == "down" else 0

    target_w = _round_up_to_64(init_img.width)
    target_h = _round_up_to_64(init_img.height + up + down)

    if up > 0:
        up = up * (target_h - init_img.height) // (up + down)
    if down > 0:
        down = target_h - init_img.height - up

    img = Image.new("RGB", (target_w, target_h))
    img.paste(init_img, ((target_w - init_img.width) // 2, up))

    mask = Image.new("L", (img.width, img.height), "white")
    draw = ImageDraw.Draw(mask)
    draw.rectangle((
        0,
        up + (mask_blur * 2 if up > 0 else 0),
        mask.width,
        mask.height - down - (mask_blur * 2 if down > 0 else 0),
    ), fill="black")

    latent_mask = Image.new("L", (img.width, img.height), "white")
    latent_draw = ImageDraw.Draw(latent_mask)
    latent_draw.rectangle((
        0,
        up + (mask_blur // 2 if up > 0 else 0),
        latent_mask.width,
        latent_mask.height - down - (mask_blur // 2 if down > 0 else 0),
    ), fill="black")

    devices.torch_gc()

    grid = images.split_grid(img, tile_w=p.width, tile_h=p.height, overlap=expand_pixels)
    grid_mask = images.split_grid(mask, tile_w=p.width, tile_h=p.height, overlap=expand_pixels)
    grid_latent_mask = images.split_grid(latent_mask, tile_w=p.width, tile_h=p.height, overlap=expand_pixels)

    work = []
    work_mask = []
    work_latent_mask = []
    work_results = []

    for (y, h, row), (_, _, row_mask), (_, _, row_latent_mask) in zip(grid.tiles, grid_mask.tiles, grid_latent_mask.tiles):
        for tiledata, tiledata_mask, tiledata_latent_mask in zip(row, row_mask, row_latent_mask):
            tile_inside_original_vertical = y >= up and y + h <= img.height - down
            if tile_inside_original_vertical:
                continue

            work.append(tiledata[2])
            work_mask.append(tiledata_mask[2])
            work_latent_mask.append(tiledata_latent_mask[2])

    print(f"Vertical Outpainting Mk3 will process {len(work)} tiles for {pass_direction} expansion.")

    for i in range(len(work)):
        p.init_images = [work[i]]
        p.image_mask = work_mask[i]
        p.latent_mask = work_latent_mask[i]
        state.job = f"Outpainting {pass_direction} tile {i + 1} out of {len(work)}"

        processed = process_images(p)

        if initial_seed is None:
            initial_seed = processed.seed
            initial_info = processed.info

        p.seed = processed.seed + 1
        work_results += processed.images

    image_index = 0
    for y, h, row in grid.tiles:
        for tiledata in row:
            tile_inside_original_vertical = y >= up and y + h <= img.height - down
            if tile_inside_original_vertical:
                continue

            tiledata[2] = work_results[image_index] if image_index < len(work_results) else Image.new("RGB", (p.width, p.height))
            image_index += 1

    combined_image = images.combine_grid(grid)
    return combined_image.crop((0, 0, target_w, init_img.height + up + down)), initial_seed, initial_info
```

- [ ] **Step 3: Review the width behavior**

Confirm the implementation keeps the input width unchanged for normal `1024`-wide images. Because `1024` is already a multiple of `64`, `target_w` remains `1024`. For non-multiple widths, it follows the original outpainting script's multiple-of-64 processing behavior.

- [ ] **Step 4: Compile the script**

Run:

```bash
python -m py_compile scripts/vertical_outpainting_mk3.py
```

Expected: PASS with no output.

- [ ] **Step 5: Run helper tests**

Run:

```bash
pytest test/test_vertical_outpainting_mk3.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit one-pass generation**

```bash
git add scripts/vertical_outpainting_mk3.py
git commit -m "feat: add vertical outpainting pass"
```

---

### Task 4: Implement Full Script Orchestration

**Files:**
- Modify: `scripts/vertical_outpainting_mk3.py`

**Interfaces:**
- Consumes:
  - `_expand_vertical_once(p, init_img, expand_pixels, pass_direction, mask_blur)`
  - `build_shift_sequences(direction, shift_preset, pixels)`
  - `zoom_content_centered(image, zoom)`
- Produces:
  - `Script.run(p, pixels, mask_blur, inpainting_fill, direction, shift_preset, zoom, inject_continue_prompt, continue_prompt) -> Processed`

- [ ] **Step 1: Add prompt injection helper**

Add:

```python
def _append_continue_prompt(prompt, continue_prompt):
    continue_prompt = (continue_prompt or "").strip()
    if not continue_prompt:
        return prompt
    if not prompt:
        return continue_prompt
    return f"{prompt}, {continue_prompt}"
```

- [ ] **Step 2: Implement `Script.run`**

Add this method inside `class Script`:

```python
    def run(self, p, pixels, mask_blur, inpainting_fill, direction, shift_preset, zoom, inject_continue_prompt, continue_prompt):
        original_prompt = p.prompt
        original_init_images = p.init_images
        original_n_iter = p.n_iter
        original_batch_size = p.batch_size
        original_do_not_save_grid = p.do_not_save_grid
        original_do_not_save_samples = p.do_not_save_samples

        sequences = build_shift_sequences(direction, shift_preset, int(pixels))
        if not sequences:
            return Processed(p, [], p.seed, "No vertical outpainting direction selected.")

        p.extra_generation_params["Vertical Outpainting MK3 pixels"] = int(pixels)
        p.extra_generation_params["Vertical Outpainting MK3 directions"] = ", ".join(direction or [])
        p.extra_generation_params["Vertical Outpainting MK3 shift preset"] = shift_preset
        p.extra_generation_params["Vertical Outpainting MK3 zoom"] = float(zoom)
        p.extra_generation_params["Vertical Outpainting MK3 inject continue prompt"] = bool(inject_continue_prompt)

        try:
            if inject_continue_prompt:
                p.prompt = _append_continue_prompt(p.prompt, continue_prompt)

            p.mask_blur = mask_blur * 2
            p.inpainting_fill = inpainting_fill
            p.inpaint_full_res = False
            p.n_iter = 1
            p.batch_size = 1
            p.do_not_save_grid = True
            p.do_not_save_samples = True

            prepared_image = zoom_content_centered(original_init_images[0], zoom)
            state.job_count = sum(len(passes) for _, passes in sequences)

            final_images = []
            initial_seed = None
            initial_info = None

            for preset_name, passes in sequences:
                current_image = prepared_image.copy()
                for pass_index, (pass_direction, pass_pixels) in enumerate(passes):
                    state.job = f"{preset_name}: pass {pass_index + 1} out of {len(passes)}"
                    current_image, pass_seed, pass_info = _expand_vertical_once(p, current_image, pass_pixels, pass_direction, mask_blur)
                    if initial_seed is None and pass_seed is not None:
                        initial_seed = pass_seed
                        initial_info = pass_info

                final_images.append(current_image)

            if opts.samples_save:
                for image in final_images:
                    images.save_image(image, p.outpath_samples, "", initial_seed, original_prompt, opts.samples_format, info=initial_info, p=p)

            return Processed(p, final_images, initial_seed, initial_info)
        finally:
            p.prompt = original_prompt
            p.init_images = original_init_images
            p.n_iter = original_n_iter
            p.batch_size = original_batch_size
            p.do_not_save_grid = original_do_not_save_grid
            p.do_not_save_samples = original_do_not_save_samples
```

- [ ] **Step 3: Verify that only final images are returned**

Read `Script.run` and confirm that only `final_images.append(current_image)` occurs after every pass in a preset completes. There must be no appending of per-pass or per-tile images to the final output list.

- [ ] **Step 4: Verify grid behavior**

Read `Script.run` and confirm it never calls `images.image_grid`, never inserts a grid into `final_images`, and never saves to `p.outpath_grids`.

- [ ] **Step 5: Compile the script**

Run:

```bash
python -m py_compile scripts/vertical_outpainting_mk3.py
```

Expected: PASS with no output.

- [ ] **Step 6: Run helper tests**

Run:

```bash
pytest test/test_vertical_outpainting_mk3.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit orchestration**

```bash
git add scripts/vertical_outpainting_mk3.py
git commit -m "feat: orchestrate vertical outpainting presets"
```

---

### Task 5: Final Verification And Review

**Files:**
- Modify only if verification finds a defect:
  - `scripts/vertical_outpainting_mk3.py`
  - `test/test_vertical_outpainting_mk3.py`

**Interfaces:**
- Consumes: Completed implementation from Tasks 1-4.
- Produces: Verified branch ready for user testing in img2img.

- [ ] **Step 1: Run targeted pytest**

Run:

```bash
pytest test/test_vertical_outpainting_mk3.py test/test_outpainting_mk2.py -q
```

Expected: PASS.

- [ ] **Step 2: Run Python compilation**

Run:

```bash
python -m py_compile scripts/vertical_outpainting_mk3.py test/test_vertical_outpainting_mk3.py
```

Expected: PASS with no output.

- [ ] **Step 3: Run ruff on touched Python files if available**

Run:

```bash
python -m ruff check scripts/vertical_outpainting_mk3.py test/test_vertical_outpainting_mk3.py
```

Expected: PASS, or SKIP only if `ruff` is not installed in the environment.

- [ ] **Step 4: Inspect git diff**

Run:

```bash
git diff -- scripts/vertical_outpainting_mk3.py test/test_vertical_outpainting_mk3.py
```

Confirm:

- New script title is exactly `Vertical Outpainting Mk3`.
- Existing outpainting scripts are untouched.
- Preset dropdown contains exactly `All`, `Centered`, `Top`, `Bottom`, `Top Centered`, and `Bottom Centered`.
- Direction UI contains only `up` and `down`.
- Pixels slider max is `384`.
- No grid is returned or saved.
- Prompt is restored in a `finally` block.

- [ ] **Step 5: Commit final fixes if needed**

If Step 4 required changes:

```bash
git add scripts/vertical_outpainting_mk3.py test/test_vertical_outpainting_mk3.py
git commit -m "fix: polish vertical outpainting mk3"
```

If no fixes are needed, do not create an empty commit.

---

## Self-Review

Spec coverage:

- New script instead of modifying existing scripts: Task 2 creates `scripts/vertical_outpainting_mk3.py`; Task 5 verifies existing scripts are untouched.
- Vertical-only UI: Task 2 adds only `up` and `down`.
- Pixels max `384`: Task 2 UI step.
- Mask blur and masked content matching Poor man's outpainting: Task 2 UI and Task 3/4 processing.
- Shift presets and direction filtering: Task 1 tests and Task 2 helper implementation.
- Sequential passes: Task 3 one-pass helper and Task 4 preset loop.
- Same total growth for presets: Task 1 expected sequences.
- Zoom-in content preprocessing: Task 1 zoom test and Task 2 helper.
- Continue prompt injection: Task 4 helper and restoration in `finally`.
- Final images only and no grid: Task 4 implementation and Task 5 diff inspection.
- Metadata: Task 4 sets `p.extra_generation_params`.
- Verification: Task 5.

Marker audit clean.

Type consistency: helper names and signatures are consistent across all tasks.
