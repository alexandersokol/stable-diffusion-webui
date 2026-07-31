# Vertical Outpainting Soft Seam Blend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic post-generation soft seam blending to `Vertical Outpainting Mk3`, `Vertical Outpainting Mk4`, and `Vertical Outpainting Mk5`.

**Architecture:** Each script keeps a small local seam-blend helper so it can remain self-contained and testable without introducing a shared module dependency for three scripts. Mk4 and Mk5 blend at final source-restore boundaries; Mk3 blends after each sequential expansion pass so later passes do not inherit hard seams. The feature is controlled by a `Soft seam blend` slider whose value is recorded in each script's metadata.

**Tech Stack:** Python, PIL/Pillow `Image`, Gradio script UI controls, existing Stable Diffusion WebUI `Processed`/`process_images`, pytest.

## Global Constraints

- Apply to `scripts/vertical_outpainting_mk3.py`, `scripts/vertical_outpainting_mk4.py`, and `scripts/vertical_outpainting_mk5.py`.
- Do not modify `poor_mans_outpainting.py` or `outpainting_mk_2.py`.
- Do not add a second inpainting pass.
- Do not add new sampler controls, ControlNet/IP-Adapter wiring, or new output variants.
- Add `Soft seam blend` slider to each script with range `0..128`, step `8`, default `32`.
- `Soft seam blend = 0` must preserve current hard-paste behavior.
- Soft seam blend is post-processing after generation and before save/return.
- Leave protected source interior exact outside the blend band.
- Leave generated gap pixels untouched outside the blend band.
- Clamp blend width to available source area and final image bounds.
- Mk5 source-only / scale-one path must not be altered.
- Record metadata keys:
  - `Vertical Outpainting MK3 soft seam blend`
  - `Vertical Outpainting MK4 soft seam blend`
  - `Vertical Outpainting MK5 soft seam blend`

---

## File Structure

- `scripts/vertical_outpainting_mk5.py`: add `soft_blend_vertical_seams(...)`, update `restore_prepared_source(...)`, UI, metadata, and `run(...)` signature.
- `test/test_vertical_outpainting_mk5.py`: add helper tests and update run-call signatures.
- `scripts/vertical_outpainting_mk4.py`: add the same local blend helper, update `restore_source_region(...)`, UI, metadata, and `run(...)` signature.
- `test/test_vertical_outpainting_mk4.py`: add helper tests and update run-call signatures.
- `scripts/vertical_outpainting_mk3.py`: add local blend helper, update `_expand_vertical_once(...)`, UI, metadata, and `run(...)` signature.
- `test/test_vertical_outpainting_mk3.py`: add helper/pass tests and update run-call signatures.

---

### Task 1: Add Mk5 Soft Seam Blend Helper And Tests

**Files:**
- Modify: `test/test_vertical_outpainting_mk5.py`
- Modify: `scripts/vertical_outpainting_mk5.py`

**Interfaces:**
- Consumes:
  - `restore_prepared_source(generated, prepared_source, source_top) -> Image.Image`
- Produces:
  - `soft_blend_vertical_seams(generated: Image.Image, source: Image.Image, source_top: int, blend_width: int) -> Image.Image`
  - `restore_prepared_source(generated, prepared_source, source_top, soft_seam_blend=0) -> Image.Image`

- [ ] **Step 1: Add failing helper tests**

Append to `test/test_vertical_outpainting_mk5.py` near the existing `test_restore_prepared_source_pastes_exact_pixels` test:

```python
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
    source = Image.new("RGB", (1, 4), "red")
    generated = Image.new("RGB", (1, 8), "blue")

    blended = mk5.soft_blend_vertical_seams(generated, source, 2, 2)

    assert blended.getpixel((0, 0)) == (0, 0, 255)
    assert blended.getpixel((0, 7)) == (0, 0, 255)
    assert blended.getpixel((0, 3)) == (255, 0, 0)
    assert blended.getpixel((0, 4)) == (255, 0, 0)
    assert blended.getpixel((0, 2)) not in [(255, 0, 0), (0, 0, 255)]
    assert blended.getpixel((0, 5)) not in [(255, 0, 0), (0, 0, 255)]
```

- [ ] **Step 2: Run tests and verify the helper is missing**

Run:

```powershell
D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_vertical_outpainting_mk5.py::test_soft_blend_vertical_seams_zero_preserves_hard_paste_result test/test_vertical_outpainting_mk5.py::test_soft_blend_vertical_seams_feathers_only_boundary_band -q
```

Expected: FAIL with `AttributeError: module '_test_vertical_outpainting_mk5' has no attribute 'soft_blend_vertical_seams'`.

- [ ] **Step 3: Implement Mk5 blend helper**

Add this helper to `scripts/vertical_outpainting_mk5.py` after `restore_prepared_source(...)`:

```python
def soft_blend_vertical_seams(generated, source, source_top, blend_width):
    result = generated.copy().convert("RGB")
    source = source.convert("RGB")
    source_top = int(source_top)
    blend_width = max(0, min(int(blend_width), source.height, result.height))
    if blend_width <= 0:
        result.paste(source, (0, source_top))
        return result

    source_bottom = source_top + source.height
    result.paste(source, (0, source_top))

    if source_top > 0:
        width = min(blend_width, source.height, source_top, result.height - source_top)
        for index in range(width):
            alpha = (index + 1) / (width + 1)
            generated_row = generated.crop((0, source_top + index, source.width, source_top + index + 1)).convert("RGB")
            source_row = source.crop((0, index, source.width, index + 1))
            result.paste(Image.blend(generated_row, source_row, alpha), (0, source_top + index))

    if source_bottom < result.height:
        width = min(blend_width, source.height, source_bottom, result.height - source_bottom)
        for index in range(width):
            alpha = 1.0 - ((index + 1) / (width + 1))
            source_y = source.height - width + index
            result_y = source_bottom - width + index
            generated_row = generated.crop((0, result_y, source.width, result_y + 1)).convert("RGB")
            source_row = source.crop((0, source_y, source.width, source_y + 1))
            result.paste(Image.blend(generated_row, source_row, alpha), (0, result_y))

    return result
```

Update `restore_prepared_source(...)`:

```python
def restore_prepared_source(generated, prepared_source, source_top, soft_seam_blend=0):
    if int(soft_seam_blend) > 0:
        return soft_blend_vertical_seams(generated, prepared_source, source_top, soft_seam_blend)

    result = generated.copy()
    result.paste(prepared_source.convert("RGB"), (0, int(source_top)))
    return result
```

- [ ] **Step 4: Run helper tests**

Run:

```powershell
D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_vertical_outpainting_mk5.py::test_soft_blend_vertical_seams_zero_preserves_hard_paste_result test/test_vertical_outpainting_mk5.py::test_soft_blend_vertical_seams_feathers_only_boundary_band -q
```

Expected: PASS.

- [ ] **Step 5: Add Mk5 UI and run integration tests**

Update all existing Mk5 `Script().run(...)` calls in `test/test_vertical_outpainting_mk5.py` to include `soft_seam_blend` between `mask_blur` and `inpainting_fill`.

Example replacement:

```python
result = mk5.Script().run(p, 8, 0.0, "Center", 1, 4, 32, 0, False, "")
```

Add:

```python
def test_run_records_soft_seam_blend_metadata_and_applies_blend():
    mk5 = load_mk5_module_for_test()
    mk5.Processed = FakeProcessed
    mk5.process_images = lambda p: FakeProcessed(p, [Image.new("RGB", (p.width, p.height), "blue")], 101, "info-101")
    mk5.opts = types.SimpleNamespace(samples_save=False, save_incomplete_images=False, samples_format="png")
    mk5.state = types.SimpleNamespace(job="", job_count=0, interrupted=False, skipped=False)
    p = make_p(Image.new("RGB", (1, 4), "red"))

    result = mk5.Script().run(p, 8, 0.0, "Center", 1, 4, 2, 0, False, "")

    assert result.images[0].getpixel((0, 2)) not in [(255, 0, 0), (0, 0, 255)]
    assert result.images[0].getpixel((0, 3)) == (255, 0, 0)
    assert p.extra_generation_params["Vertical Outpainting MK5 soft seam blend"] == 2
```

Run:

```powershell
D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_vertical_outpainting_mk5.py -q
```

Expected before implementation: FAIL with `TypeError` or missing metadata.

- [ ] **Step 6: Implement Mk5 UI and run integration**

In `scripts/vertical_outpainting_mk5.py`, add the slider after `mask_blur`:

```python
soft_seam_blend = gr.Slider(label="Soft seam blend", minimum=0, maximum=128, step=8, value=32, elem_id=self.elem_id("soft_seam_blend"))
```

Return it between `mask_blur` and `inpainting_fill`.

Change the run signature:

```python
def run(self, p, target_height, scale, placement, seam_size, mask_blur, soft_seam_blend, inpainting_fill, inject_continue_prompt, continue_prompt):
```

Record metadata:

```python
p.extra_generation_params["Vertical Outpainting MK5 soft seam blend"] = int(soft_seam_blend)
```

Keep source-only path unchanged except for metadata; do not call `soft_blend_vertical_seams` when `float(scale) == 1.0 or not mask_has_white(mask)`.

Update generated path:

```python
final_image = restore_prepared_source(generated_image, prepared_source, source_top, soft_seam_blend)
```

Run:

```powershell
D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_vertical_outpainting_mk5.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit Mk5 seam blend**

```powershell
git add scripts/vertical_outpainting_mk5.py test/test_vertical_outpainting_mk5.py
git commit -m "feat: add soft seam blend to vertical outpainting mk5"
```

---

### Task 2: Add Mk4 Soft Seam Blend

**Files:**
- Modify: `test/test_vertical_outpainting_mk4.py`
- Modify: `scripts/vertical_outpainting_mk4.py`

**Interfaces:**
- Consumes:
  - `soft_blend_vertical_seams(generated, source, source_top, blend_width) -> Image.Image` pattern from Task 1.
- Produces:
  - `restore_source_region(generated, source, source_top, source_handling, seam_size, soft_seam_blend=0) -> Image.Image`
  - `Script.run(p, target_height, source_placement, source_handling, seam_size, mask_blur, soft_seam_blend, inpainting_fill, inject_continue_prompt, continue_prompt) -> Processed`

- [ ] **Step 1: Add Mk4 helper tests**

Append near the existing `restore_source_region` tests:

```python
def test_restore_source_region_preserve_with_soft_blend_feathers_edges_only():
    mk4 = load_mk4_module_for_test()
    source = Image.new("RGB", (1, 4), "red")
    generated = Image.new("RGB", (1, 8), "blue")

    restored = mk4.restore_source_region(generated, source, 2, "Preserve source pixels", 1, 2)

    assert restored.getpixel((0, 0)) == (0, 0, 255)
    assert restored.getpixel((0, 7)) == (0, 0, 255)
    assert restored.getpixel((0, 2)) not in [(255, 0, 0), (0, 0, 255)]
    assert restored.getpixel((0, 3)) == (255, 0, 0)
    assert restored.getpixel((0, 4)) == (255, 0, 0)
    assert restored.getpixel((0, 5)) not in [(255, 0, 0), (0, 0, 255)]


def test_restore_source_region_soft_resynthesis_ignores_soft_blend():
    mk4 = load_mk4_module_for_test()
    source = Image.new("RGB", (1, 4), "red")
    generated = Image.new("RGB", (1, 8), "blue")

    restored = mk4.restore_source_region(generated, source, 2, "Soft resynthesis", 1, 2)

    assert restored.tobytes() == generated.tobytes()
```

Run:

```powershell
D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_vertical_outpainting_mk4.py::test_restore_source_region_preserve_with_soft_blend_feathers_edges_only test/test_vertical_outpainting_mk4.py::test_restore_source_region_soft_resynthesis_ignores_soft_blend -q
```

Expected: FAIL because `restore_source_region(...)` does not accept `soft_seam_blend`.

- [ ] **Step 2: Implement Mk4 helper and restore behavior**

Add `soft_blend_vertical_seams(...)` to `scripts/vertical_outpainting_mk4.py` using the same implementation from Task 1.

Change `restore_source_region(...)` signature:

```python
def restore_source_region(generated, source, source_top, source_handling, seam_size, soft_seam_blend=0):
```

For `source_handling == "Soft resynthesis"`, continue returning `generated.copy()`.

For `source_handling == "Preserve source pixels"`:

```python
if int(soft_seam_blend) > 0:
    return soft_blend_vertical_seams(generated, source, source_top, soft_seam_blend)
result.paste(source, (0, source_top))
return result
```

For `source_handling == "Blend seam only"`, preserve existing protected-inner-source behavior, then apply soft blending only if `soft_seam_blend > 0`:

```python
if int(soft_seam_blend) > 0:
    result = soft_blend_vertical_seams(result, source, source_top, soft_seam_blend)
```

Run:

```powershell
D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_vertical_outpainting_mk4.py::test_restore_source_region_preserve_with_soft_blend_feathers_edges_only test/test_vertical_outpainting_mk4.py::test_restore_source_region_soft_resynthesis_ignores_soft_blend -q
```

Expected: PASS.

- [ ] **Step 3: Add Mk4 UI/run tests**

Update all existing Mk4 `Script().run(...)` calls in `test/test_vertical_outpainting_mk4.py` to include `soft_seam_blend` between `mask_blur` and `inpainting_fill`.

Example:

```python
result = mk4.Script().run(p, 8, "All", "Preserve source pixels", 1, 4, 32, 0, False, "")
```

Add metadata assertion to `test_run_records_mk4_generation_metadata_and_respects_explicit_placement`:

```python
assert p.extra_generation_params["Vertical Outpainting MK4 soft seam blend"] == 32
```

Run:

```powershell
D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_vertical_outpainting_mk4.py -q
```

Expected before implementation: FAIL due to run signature mismatch or missing metadata.

- [ ] **Step 4: Implement Mk4 UI/run integration**

Add slider after `mask_blur`:

```python
soft_seam_blend = gr.Slider(label="Soft seam blend", minimum=0, maximum=128, step=8, value=32, elem_id=self.elem_id("soft_seam_blend"))
```

Return it between `mask_blur` and `inpainting_fill`.

Change run signature:

```python
def run(self, p, target_height, source_placement, source_handling, seam_size, mask_blur, soft_seam_blend, inpainting_fill, inject_continue_prompt, continue_prompt):
```

Record metadata:

```python
p.extra_generation_params["Vertical Outpainting MK4 soft seam blend"] = int(soft_seam_blend)
```

Update restore call:

```python
final_image = restore_source_region(generated, source, source_top, source_handling, seam_size, soft_seam_blend)
```

Run:

```powershell
D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_vertical_outpainting_mk4.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Mk4 seam blend**

```powershell
git add scripts/vertical_outpainting_mk4.py test/test_vertical_outpainting_mk4.py
git commit -m "feat: add soft seam blend to vertical outpainting mk4"
```

---

### Task 3: Add Mk3 Soft Seam Blend

**Files:**
- Modify: `test/test_vertical_outpainting_mk3.py`
- Modify: `scripts/vertical_outpainting_mk3.py`

**Interfaces:**
- Consumes:
  - Mk3 `_expand_vertical_once(p, init_img, expand_pixels, pass_direction, mask_blur)` current behavior.
- Produces:
  - `soft_blend_vertical_seams(generated, source, source_top, blend_width) -> Image.Image`
  - `_expand_vertical_once(p, init_img, expand_pixels, pass_direction, mask_blur, soft_seam_blend=0) -> tuple[Image.Image, int, str]`
  - `Script.run(p, pixels, mask_blur, soft_seam_blend, inpainting_fill, direction, shift_preset, zoom, inject_continue_prompt, continue_prompt) -> Processed`

- [ ] **Step 1: Add Mk3 pass-level blend tests**

Append to `test/test_vertical_outpainting_mk3.py`:

```python
def test_soft_blend_vertical_seams_mk3_zero_preserves_hard_paste():
    mk3 = load_mk3_module_for_test()
    source = Image.new("RGB", (1, 4), "red")
    generated = Image.new("RGB", (1, 8), "blue")

    blended = mk3.soft_blend_vertical_seams(generated, source, 2, 0)

    expected = generated.copy()
    expected.paste(source, (0, 2))
    assert blended.tobytes() == expected.tobytes()


def test_soft_blend_vertical_seams_mk3_feathers_only_boundary_band():
    mk3 = load_mk3_module_for_test()
    source = Image.new("RGB", (1, 4), "red")
    generated = Image.new("RGB", (1, 8), "blue")

    blended = mk3.soft_blend_vertical_seams(generated, source, 2, 2)

    assert blended.getpixel((0, 0)) == (0, 0, 255)
    assert blended.getpixel((0, 7)) == (0, 0, 255)
    assert blended.getpixel((0, 2)) not in [(255, 0, 0), (0, 0, 255)]
    assert blended.getpixel((0, 3)) == (255, 0, 0)
    assert blended.getpixel((0, 4)) == (255, 0, 0)
    assert blended.getpixel((0, 5)) not in [(255, 0, 0), (0, 0, 255)]
```

Run:

```powershell
D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_vertical_outpainting_mk3.py::test_soft_blend_vertical_seams_mk3_zero_preserves_hard_paste test/test_vertical_outpainting_mk3.py::test_soft_blend_vertical_seams_mk3_feathers_only_boundary_band -q
```

Expected: FAIL with missing helper.

- [ ] **Step 2: Implement Mk3 helper**

Add `soft_blend_vertical_seams(...)` to `scripts/vertical_outpainting_mk3.py` using the same implementation from Task 1.

Run:

```powershell
D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_vertical_outpainting_mk3.py::test_soft_blend_vertical_seams_mk3_zero_preserves_hard_paste test/test_vertical_outpainting_mk3.py::test_soft_blend_vertical_seams_mk3_feathers_only_boundary_band -q
```

Expected: PASS.

- [ ] **Step 3: Add Mk3 `_expand_vertical_once` blend coverage**

Add this test:

```python
def test_expand_vertical_once_soft_blends_current_image_after_generation():
    mk3 = load_mk3_module_for_test()
    calls = []

    class FakeProcessed:
        def __init__(self, _p, images, seed, info, **kwargs):
            self.images = images
            self.seed = seed
            self.info = info

    def fake_process(p):
        calls.append(p.init_images[0].size)
        return FakeProcessed(p, [Image.new("RGB", p.init_images[0].size, "blue")], 101, "info-101")

    mk3.process_images = fake_process
    p = types.SimpleNamespace(
        init_images=[],
        image_mask=None,
        latent_mask=None,
        width=4,
        height=4,
        seed=42,
    )

    result, seed, info = mk3._expand_vertical_once(p, Image.new("RGB", (4, 4), "red"), 4, "up", 0, 2)

    assert calls
    assert seed == 101
    assert info == "info-101"
    assert result.size == (4, 8)
    assert result.getpixel((0, 4)) not in [(255, 0, 0), (0, 0, 255)]
    assert result.getpixel((0, 5)) == (255, 0, 0)
```

Run:

```powershell
D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_vertical_outpainting_mk3.py::test_expand_vertical_once_soft_blends_current_image_after_generation -q
```

Expected before implementation: FAIL because `_expand_vertical_once(...)` does not accept the extra argument.

- [ ] **Step 4: Update Mk3 pass implementation**

Change signature:

```python
def _expand_vertical_once(p, init_img, expand_pixels, pass_direction, mask_blur, soft_seam_blend=0):
```

After:

```python
combined_image = images.combine_grid(grid)
```

replace the return with:

```python
final = combined_image.crop((0, 0, logical_w, logical_h))
final = soft_blend_vertical_seams(final, init_img, up, soft_seam_blend)
return final, initial_seed, initial_info
```

Run:

```powershell
D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_vertical_outpainting_mk3.py::test_expand_vertical_once_soft_blends_current_image_after_generation -q
```

Expected: PASS.

- [ ] **Step 5: Add Mk3 UI/run integration**

Update all existing Mk3 `Script().run(...)` calls in `test/test_vertical_outpainting_mk3.py` to include `soft_seam_blend` between `mask_blur` and `inpainting_fill`.

Example:

```python
result = mk3.Script().run(p, 8, 4, 32, 0, ["up", "down"], "All", 1.0, False, "")
```

In the fake `expand_once` test helper, update signature:

```python
def expand_once(_p, image, _pixels, _direction, _mask_blur, _soft_seam_blend):
```

Add metadata assertion:

```python
assert p.extra_generation_params["Vertical Outpainting MK3 soft seam blend"] == 32
```

Run:

```powershell
D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_vertical_outpainting_mk3.py -q
```

Expected before implementation: FAIL due to run signature mismatch or missing metadata.

- [ ] **Step 6: Implement Mk3 UI/run integration**

Add slider after `mask_blur`:

```python
soft_seam_blend = gr.Slider(label="Soft seam blend", minimum=0, maximum=128, step=8, value=32, elem_id=self.elem_id("soft_seam_blend"))
```

Return it between `mask_blur` and `inpainting_fill`.

Change run signature:

```python
def run(self, p, pixels, mask_blur, soft_seam_blend, inpainting_fill, direction, shift_preset, zoom, inject_continue_prompt, continue_prompt):
```

Record metadata:

```python
p.extra_generation_params["Vertical Outpainting MK3 soft seam blend"] = int(soft_seam_blend)
```

Update pass call:

```python
current_image, pass_seed, pass_info = _expand_vertical_once(p, current_image, pass_pixels, pass_direction, mask_blur, soft_seam_blend)
```

Run:

```powershell
D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_vertical_outpainting_mk3.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit Mk3 seam blend**

```powershell
git add scripts/vertical_outpainting_mk3.py test/test_vertical_outpainting_mk3.py
git commit -m "feat: add soft seam blend to vertical outpainting mk3"
```

---

### Task 4: Regression Verification And Final Polish

**Files:**
- Verify: `scripts/vertical_outpainting_mk3.py`
- Verify: `scripts/vertical_outpainting_mk4.py`
- Verify: `scripts/vertical_outpainting_mk5.py`
- Verify: `test/test_vertical_outpainting_mk3.py`
- Verify: `test/test_vertical_outpainting_mk4.py`
- Verify: `test/test_vertical_outpainting_mk5.py`

**Interfaces:**
- Consumes complete Mk3/Mk4/Mk5 seam blend implementations.
- Produces final verification evidence.

- [ ] **Step 1: Run focused Mk3/Mk4/Mk5 tests**

Run:

```powershell
D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_vertical_outpainting_mk3.py test/test_vertical_outpainting_mk4.py test/test_vertical_outpainting_mk5.py -q
```

Expected: PASS. Existing `PytestConfigWarning: Unknown config option: base_url` is acceptable.

- [ ] **Step 2: Run combined outpainting regression tests**

Run:

```powershell
D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_vertical_outpainting_mk5.py test/test_vertical_outpainting_mk4.py test/test_vertical_outpainting_mk3.py test/test_outpainting_mk2.py -q
```

Expected: PASS. Existing `PytestConfigWarning: Unknown config option: base_url` is acceptable.

- [ ] **Step 3: Compile edited scripts and tests**

Run:

```powershell
D:\stable-diffusion\envs\v1101\python.exe -m py_compile scripts/vertical_outpainting_mk3.py scripts/vertical_outpainting_mk4.py scripts/vertical_outpainting_mk5.py test/test_vertical_outpainting_mk3.py test/test_vertical_outpainting_mk4.py test/test_vertical_outpainting_mk5.py
```

Expected: no output and exit code `0`.

- [ ] **Step 4: Attempt ruff if available**

Run:

```powershell
D:\stable-diffusion\envs\v1101\python.exe -m ruff check scripts/vertical_outpainting_mk3.py scripts/vertical_outpainting_mk4.py scripts/vertical_outpainting_mk5.py test/test_vertical_outpainting_mk3.py test/test_vertical_outpainting_mk4.py test/test_vertical_outpainting_mk5.py
```

Expected: pass if ruff is installed. If it fails with `No module named ruff`, record that ruff is unavailable and do not install dependencies.

- [ ] **Step 5: Inspect UI signatures**

Run:

```powershell
rg -n "Soft seam blend|soft_seam_blend|Vertical Outpainting MK[345] soft seam blend|def run\\(" scripts/vertical_outpainting_mk3.py scripts/vertical_outpainting_mk4.py scripts/vertical_outpainting_mk5.py test/test_vertical_outpainting_mk3.py test/test_vertical_outpainting_mk4.py test/test_vertical_outpainting_mk5.py
```

Expected: each script has one slider, one metadata key, and all test call sites include the new argument.

- [ ] **Step 6: Commit verification note if needed**

If no code changes are needed in this task, do not create an empty commit. Record verification evidence in the implementation summary instead.

If small polish fixes are needed, commit them:

```powershell
git add scripts/vertical_outpainting_mk3.py scripts/vertical_outpainting_mk4.py scripts/vertical_outpainting_mk5.py test/test_vertical_outpainting_mk3.py test/test_vertical_outpainting_mk4.py test/test_vertical_outpainting_mk5.py
git commit -m "test: verify vertical outpainting soft seam blend"
```
