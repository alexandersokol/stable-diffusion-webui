# SD WebUI RunPod Memory Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce RunPod system RAM growth during large Dynamic Prompts batches by freeing CPU heap after generation and, when explicitly enabled, avoiding retention of full saved `PIL.Image` results that will not be shown in the gallery.

**Architecture:** Implement this in two layers. First add a conservative CPU cleanup helper called after Gradio GPU jobs complete, so completed large jobs have a chance to return memory to the OS. Then finish the existing `modules.processing_output` save-and-placeholder path behind a new opt-in setting, so RunPod batch profiles can reduce peak RAM without changing default upstream-like behavior.

**Tech Stack:** Python, Pillow `Image`, pytest, A1111 options via `modules/shared_options.py`, existing WebUI processing flow in `modules/processing.py`, `modules/call_queue.py`, and `modules/devices.py`.

**Spec:** `D:\stable-diffusion\sd-webui-memory-leak-findings.md`

## Global Constraints

- Do not roll back unrelated local changes; current working tree already contains in-progress changes in `modules/processing_output.py` and `test/test_processing_memory.py`.
- Default behavior must remain upstream-like: new save-and-drop behavior is disabled unless explicitly enabled.
- API behavior must remain compatible: `p.is_api=True` keeps full images so base64/file response modes continue to work.
- Grid behavior must remain compatible: if `return_grid` or `grid_save` is enabled, keep full images until the grid is built.
- Infotexts must be preserved for every generated image.
- Saved image paths must continue to be available through `already_saved_as` placeholders.
- CPU malloc trimming must be best-effort and platform-safe: call glibc `malloc_trim(0)` only on Linux when available.
- Avoid adding third-party dependencies.

## Review Focus

- Dynamic Prompts-style huge txt2img run with `do_not_show_images=true`, `save_samples=true`, `return_grid=false`, `grid_save=false`, and new drop option enabled: output should use 1x1 placeholders, preserve saved paths, and close the full images.
- Normal UI run with the new drop option disabled: full images should still be retained through `Processed.images`.
- Grid generation with the new drop option enabled: full images must remain available until grid creation completes.
- API generation with the new drop option enabled: full images must remain available for API encoding.
- Non-Linux hosts and Linux hosts without `malloc_trim`: CPU cleanup must not raise.

---

## File Structure

- Modify `modules/shared_options.py`: add opt-in settings near Gallery or Stable Diffusion memory options.
- Modify `modules/devices.py`: add reusable CPU cleanup helpers next to existing `torch_gc`.
- Modify `modules/call_queue.py`: call CPU cleanup after Gradio GPU calls, after `devices.torch_gc()`.
- Modify `modules/processing_output.py`: preserve existing helper shape, but require the new option before replacing full images during generation.
- Modify `modules/processing.py`: replace direct result appends in main image, mask, and mask-composite branches with `processing_output.append_image_result(...)`.
- Modify `test/test_processing_memory.py`: update existing tests for the new opt-in setting and add regression tests for disabled/default, grid, API, and missing save path cases.
- Create `test/test_devices_cpu_cleanup.py`: unit-test Python GC and Linux malloc trimming behavior without depending on actual glibc.
- Create or modify `test/test_call_queue_cleanup.py`: unit-test that queued Gradio GPU calls invoke CPU cleanup after the wrapped function returns and after exception fallback handling.

## Task 1: Add Opt-In CPU Cleanup Helper

**Files:**
- Modify: `modules/devices.py`
- Modify: `modules/shared_options.py`
- Test: `test/test_devices_cpu_cleanup.py`

**Interfaces:**
- Produces: `devices.cpu_gc(force=False) -> None`
- Produces: `devices.malloc_trim() -> bool`
- Consumes later: `modules.call_queue.wrap_gradio_call_no_job` calls `devices.cpu_gc(force=True)` after GPU calls.

- [ ] **Step 1: Write failing tests for CPU cleanup**

Create `test/test_devices_cpu_cleanup.py`:

```python
import types

from modules import devices


def test_cpu_gc_runs_python_gc_when_enabled(monkeypatch):
    calls = []
    monkeypatch.setattr(devices.shared, "opts", types.SimpleNamespace(cpu_gc_after_generation=True, cpu_gc_malloc_trim=True), raising=False)
    monkeypatch.setattr(devices.gc, "collect", lambda: calls.append("collect"))
    monkeypatch.setattr(devices, "malloc_trim", lambda: calls.append("trim") or True)

    devices.cpu_gc(force=True)

    assert calls == ["collect", "trim"]


def test_cpu_gc_skips_when_disabled_without_force(monkeypatch):
    calls = []
    monkeypatch.setattr(devices.shared, "opts", types.SimpleNamespace(cpu_gc_after_generation=False, cpu_gc_malloc_trim=True), raising=False)
    monkeypatch.setattr(devices.gc, "collect", lambda: calls.append("collect"))
    monkeypatch.setattr(devices, "malloc_trim", lambda: calls.append("trim") or True)

    devices.cpu_gc()

    assert calls == []


def test_malloc_trim_returns_false_on_non_linux(monkeypatch):
    monkeypatch.setattr(devices.sys, "platform", "win32")

    assert devices.malloc_trim() is False


def test_malloc_trim_returns_false_when_libc_unavailable(monkeypatch):
    monkeypatch.setattr(devices.sys, "platform", "linux")
    monkeypatch.setattr(devices.ctypes, "CDLL", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("missing libc")))

    assert devices.malloc_trim() is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest test/test_devices_cpu_cleanup.py -q`

Expected: FAIL because `devices.cpu_gc`, `devices.malloc_trim`, `devices.gc`, or `devices.ctypes` do not exist yet.

- [ ] **Step 3: Add options**

In `modules/shared_options.py`, near existing memory-related options in the Stable Diffusion section or near Gallery if that is easier to discover, add:

```python
"cpu_gc_after_generation": OptionInfo(False, "Run Python CPU garbage collection after generation"),
"cpu_gc_malloc_trim": OptionInfo(True, "Try to return freed CPU memory to the OS after CPU garbage collection").info("Linux/glibc only"),
```

- [ ] **Step 4: Implement helpers**

In `modules/devices.py`, add imports at the top:

```python
import ctypes
import gc
```

Add below `torch_gc`:

```python
def malloc_trim():
    if sys.platform != "linux":
        return False

    try:
        libc = ctypes.CDLL("libc.so.6")
        return libc.malloc_trim(0) == 1
    except Exception:
        return False


def cpu_gc(force=False):
    if not force and not getattr(shared.opts, "cpu_gc_after_generation", False):
        return

    gc.collect()

    if getattr(shared.opts, "cpu_gc_malloc_trim", True):
        malloc_trim()
```

- [ ] **Step 5: Run CPU cleanup tests**

Run: `python -m pytest test/test_devices_cpu_cleanup.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add modules/devices.py modules/shared_options.py test/test_devices_cpu_cleanup.py
git commit -m "feat: add optional CPU cleanup after generation"
```

## Task 2: Invoke CPU Cleanup After Gradio GPU Jobs

**Files:**
- Modify: `modules/call_queue.py`
- Test: `test/test_call_queue_cleanup.py`

**Interfaces:**
- Consumes: `devices.cpu_gc(force=True) -> None`
- Produces: wrapped Gradio GPU calls run CPU cleanup after `devices.torch_gc()`.

- [ ] **Step 1: Write failing tests**

Create `test/test_call_queue_cleanup.py`:

```python
from modules import call_queue


def test_gradio_call_runs_cpu_gc_after_success(monkeypatch):
    calls = []
    monkeypatch.setattr(call_queue.devices, "torch_gc", lambda: calls.append("torch"))
    monkeypatch.setattr(call_queue.devices, "cpu_gc", lambda force=False: calls.append(("cpu", force)))

    wrapped = call_queue.wrap_gradio_call_no_job(lambda: ("image", "info"), add_stats=False)

    assert wrapped() == ("image", "info")
    assert calls == ["torch", ("cpu", True)]


def test_gradio_call_runs_cpu_gc_after_exception_fallback(monkeypatch):
    calls = []
    monkeypatch.setattr(call_queue.devices, "torch_gc", lambda: calls.append("torch"))
    monkeypatch.setattr(call_queue.devices, "cpu_gc", lambda force=False: calls.append(("cpu", force)))
    monkeypatch.setattr(call_queue.errors, "report", lambda *_args, **_kwargs: None)

    def raises():
        raise RuntimeError("boom")

    wrapped = call_queue.wrap_gradio_call_no_job(raises, extra_outputs=[None, ""], add_stats=False)

    result = wrapped()

    assert result[0:2] == (None, "")
    assert "RuntimeError" in result[-1]
    assert calls == ["torch", ("cpu", True)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest test/test_call_queue_cleanup.py -q`

Expected: FAIL because `wrap_gradio_call_no_job` does not call `devices.cpu_gc(force=True)`.

- [ ] **Step 3: Implement call site**

In `modules/call_queue.py`, directly after:

```python
devices.torch_gc()
```

add:

```python
devices.cpu_gc(force=True)
```

- [ ] **Step 4: Run call queue tests**

Run: `python -m pytest test/test_call_queue_cleanup.py -q`

Expected: PASS.

- [ ] **Step 5: Run existing GC tests**

Run: `python -m pytest test/test_devices_gc.py test/test_call_queue_cleanup.py test/test_devices_cpu_cleanup.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add modules/call_queue.py test/test_call_queue_cleanup.py
git commit -m "feat: clean CPU heap after gradio GPU calls"
```

## Task 3: Gate Save-And-Drop Behavior Behind an Explicit Option

**Files:**
- Modify: `modules/shared_options.py`
- Modify: `modules/processing_output.py`
- Modify: `test/test_processing_memory.py`

**Interfaces:**
- Produces: `processing_output.should_store_saved_image_as_placeholder(p, opts, saved_path) -> bool`
- Produces: `processing_output.append_image_result(output_images, output_image_paths, p, opts, image, saved_path) -> None`
- Consumes later: `modules.processing.process_images_inner` uses `append_image_result`.

- [ ] **Step 1: Update tests for opt-in behavior**

In `test/test_processing_memory.py`, ensure helper tests use `drop_saved_images_from_results=True` when expecting placeholder behavior:

```python
def test_append_image_result_drops_saved_hidden_non_api_image(tmp_path):
    saved_path = str(tmp_path / "saved.png")
    original = Image.new("RGB", (64, 64), "red")
    p = SimpleNamespace(is_api=False)
    opts = SimpleNamespace(
        do_not_show_images=True,
        return_grid=False,
        grid_save=False,
        drop_saved_images_from_results=True,
    )
    output_images = []
    output_image_paths = []

    processing_output.append_image_result(output_images, output_image_paths, p, opts, original, saved_path)

    assert output_images[0] is not original
    assert output_images[0].size == (1, 1)
    assert output_images[0].already_saved_as == saved_path
    assert output_image_paths == [saved_path]
```

Add the default-off regression:

```python
def test_append_image_result_keeps_hidden_image_when_drop_option_disabled(tmp_path):
    saved_path = str(tmp_path / "saved.png")
    original = Image.new("RGB", (64, 64), "red")
    p = SimpleNamespace(is_api=False)
    opts = SimpleNamespace(
        do_not_show_images=True,
        return_grid=False,
        grid_save=False,
        drop_saved_images_from_results=False,
    )
    output_images = []
    output_image_paths = []

    processing_output.append_image_result(output_images, output_image_paths, p, opts, original, saved_path)

    assert output_images == [original]
    assert output_image_paths == [saved_path]
```

Update existing grid/API tests to include `drop_saved_images_from_results=True` so they prove those safety exits still win.

- [ ] **Step 2: Run tests to verify the new default-off test fails**

Run: `python -m pytest test/test_processing_memory.py -q`

Expected: FAIL until `should_store_saved_image_as_placeholder` checks the new option.

- [ ] **Step 3: Add the setting**

In `modules/shared_options.py`, inside `options_section(('ui_gallery', "Gallery", "ui"), { ... })`, after `do_not_show_images`, add:

```python
"drop_saved_images_from_results": OptionInfo(False, "Drop saved images from generation results when gallery is hidden").info("reduces RAM for large saved batches; disables by default for compatibility"),
```

- [ ] **Step 4: Gate placeholder storage**

In `modules/processing_output.py`, update `should_store_saved_image_as_placeholder`:

```python
def should_store_saved_image_as_placeholder(p, opts, saved_path):
    if not saved_path:
        return False

    if getattr(p, "is_api", False):
        return False

    if not getattr(opts, "drop_saved_images_from_results", False):
        return False

    if not getattr(opts, "do_not_show_images", False):
        return False

    return not getattr(opts, "return_grid", False) and not getattr(opts, "grid_save", False)
```

- [ ] **Step 5: Run processing memory tests**

Run: `python -m pytest test/test_processing_memory.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add modules/shared_options.py modules/processing_output.py test/test_processing_memory.py
git commit -m "feat: gate saved image placeholders behind option"
```

## Task 4: Use Save-And-Drop In The Main Processing Hot Path

**Files:**
- Modify: `modules/processing.py`
- Modify: `test/test_processing_memory.py`

**Interfaces:**
- Consumes: `processing_output.append_image_result(output_images, output_image_paths, p, opts, image, saved_path)`
- Produces: main generation path no longer appends full saved images when the new drop mode is enabled and no grid is required.

- [ ] **Step 1: Add focused integration test for the helper contract**

In `test/test_processing_memory.py`, add a mask-safe helper test that represents the repeated append sites in `processing.py`:

```python
def test_append_image_result_keeps_unsaved_hidden_image_even_when_drop_enabled():
    original = Image.new("RGB", (64, 64), "red")
    p = SimpleNamespace(is_api=False)
    opts = SimpleNamespace(
        do_not_show_images=True,
        return_grid=False,
        grid_save=False,
        drop_saved_images_from_results=True,
    )
    output_images = []
    output_image_paths = []

    processing_output.append_image_result(output_images, output_image_paths, p, opts, original, None)

    assert output_images == [original]
    assert output_image_paths == [None]
```

- [ ] **Step 2: Run helper tests**

Run: `python -m pytest test/test_processing_memory.py -q`

Expected: PASS if Task 3 is complete.

- [ ] **Step 3: Replace main image append**

In `modules/processing.py`, replace:

```python
output_images.append(image)
output_image_paths.append(saved_image_path)
```

with:

```python
processing_output.append_image_result(output_images, output_image_paths, p, opts, image, saved_image_path)
```

- [ ] **Step 4: Replace returned mask append**

In the `if opts.return_mask:` branch, replace:

```python
output_images.append(image_mask)
output_image_paths.append(saved_mask_path)
```

with:

```python
processing_output.append_image_result(output_images, output_image_paths, p, opts, image_mask, saved_mask_path)
```

- [ ] **Step 5: Replace returned mask composite append**

In the `if opts.return_mask_composite:` branch, replace:

```python
output_images.append(image_mask_composite)
output_image_paths.append(saved_mask_composite_path)
```

with:

```python
processing_output.append_image_result(output_images, output_image_paths, p, opts, image_mask_composite, saved_mask_composite_path)
```

- [ ] **Step 6: Keep postprocess placeholder replacement**

Leave this existing call after script postprocessing:

```python
processing_output.replace_saved_images_with_placeholders(p, res, output_image_paths)
```

This continues to reduce retained result size after scripts have had their final `Processed` hook. During code review, explicitly check whether any script hook requires full images before this point when `drop_saved_images_from_results=True`; if so, the option should be documented as batch/headless mode only.

- [ ] **Step 7: Run targeted processing tests**

Run: `python -m pytest test/test_processing_memory.py test/test_api_image_response.py -q`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add modules/processing.py test/test_processing_memory.py
git commit -m "feat: drop saved hidden images during processing"
```

## Task 5: Preserve API And Gallery Response Behavior

**Files:**
- Modify: `test/test_api_image_response.py`
- Modify: `test/test_processing_memory.py`

**Interfaces:**
- Consumes: placeholders use `image.already_saved_as` and `size == (1, 1)`.
- Produces: confidence that API and UI file references still work.

- [ ] **Step 1: Add API placeholder file-mode test**

In `test/test_api_image_response.py`, add:

```python
def test_encode_processed_images_for_api_file_mode_accepts_placeholder_path(tmp_path):
    saved_path = str(tmp_path / "saved.png")
    image = Image.new("RGB", (1, 1), "black")
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
```

- [ ] **Step 2: Add replace helper mismatch test for extra script outputs**

In `test/test_processing_memory.py`, keep or add:

```python
def test_replace_saved_images_with_placeholders_skips_when_script_changed_image_count(tmp_path):
    saved_path = str(tmp_path / "saved.png")
    original = Image.new("RGB", (64, 64), "red")
    processed = SimpleNamespace(images=[original])
    p = SimpleNamespace(is_api=False)

    processing_output.replace_saved_images_with_placeholders(p, processed, [saved_path, None])

    assert processed.images[0] is original
```

- [ ] **Step 3: Run behavior tests**

Run: `python -m pytest test/test_api_image_response.py test/test_processing_memory.py -q`

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add test/test_api_image_response.py test/test_processing_memory.py
git commit -m "test: cover placeholder image response behavior"
```

## Task 6: Configure RunPod Defaults

**Files:**
- Modify: `D:\stable-diffusion\runpod-build\start-runpod.sh` or the image/config layer that writes `/workspace/sd/config/config.json`
- Optional modify: RunPod template environment docs if present in `D:\stable-diffusion\runpod-build`

**Interfaces:**
- Consumes: `drop_saved_images_from_results`, `cpu_gc_after_generation`, and `cpu_gc_malloc_trim`.
- Produces: RunPod image/profile enables the new batch-safe behavior without changing default local WebUI behavior.

- [ ] **Step 1: Locate config bootstrap code**

Run:

```bash
rg -n "config.json|ui-config.json|WEBUI_EXTRA_ARGS|WEBUI_MEMORY_ARGS|do_not_show_images" D:\stable-diffusion\runpod-build
```

Expected: identify the script or seed config responsible for persistent `/workspace/sd/config/config.json`.

- [ ] **Step 2: Add RunPod config defaults**

Where the RunPod image seeds `config.json`, set:

```json
"cpu_gc_after_generation": true,
"cpu_gc_malloc_trim": true,
"drop_saved_images_from_results": true,
"do_not_show_images": true,
"return_grid": false,
"grid_save": false
```

Do not set `WEBUI_MEMORY_ARGS` to `--medvram`, `--medvram-sdxl`, or `--lowvram`.

- [ ] **Step 3: Add smoke-check command to docs or launch log**

Document this RunPod verification command:

```bash
python - <<'PY'
import json
from pathlib import Path
cfg = json.loads(Path("/workspace/sd/config/config.json").read_text())
for key in ["cpu_gc_after_generation", "cpu_gc_malloc_trim", "drop_saved_images_from_results", "do_not_show_images", "return_grid", "grid_save"]:
    print(key, cfg.get(key))
PY
```

Expected output:

```text
cpu_gc_after_generation True
cpu_gc_malloc_trim True
drop_saved_images_from_results True
do_not_show_images True
return_grid False
grid_save False
```

- [ ] **Step 4: Commit**

```bash
git add D:\stable-diffusion\runpod-build
git commit -m "chore: enable batch memory cleanup in runpod config"
```

## Task 7: Verification Pass

**Files:**
- No required source files.

**Interfaces:**
- Consumes: all previous task outputs.
- Produces: evidence that the change is ready for RunPod manual validation.

- [ ] **Step 1: Run targeted unit tests**

Run:

```bash
python -m pytest test/test_processing_memory.py test/test_api_image_response.py test/test_devices_cpu_cleanup.py test/test_call_queue_cleanup.py test/test_devices_gc.py -q
```

Expected: PASS.

- [ ] **Step 2: Run broader smoke tests around generation plumbing**

Run:

```bash
python -m pytest test/test_extras.py test/test_progress_results.py test/test_shared_state.py -q
```

Expected: PASS.

- [ ] **Step 3: Inspect diff for unsafe global behavior changes**

Run:

```bash
git diff -- modules/processing.py modules/processing_output.py modules/devices.py modules/call_queue.py modules/shared_options.py test/test_processing_memory.py test/test_api_image_response.py test/test_devices_cpu_cleanup.py test/test_call_queue_cleanup.py
```

Expected: save-and-drop requires `drop_saved_images_from_results=True`; API and grid safety exits are still present.

- [ ] **Step 4: Manual RunPod validation**

On a pod using the patched image/profile:

```text
1. Confirm launch log has no --medvram, --medvram-sdxl, or --lowvram.
2. Confirm config has cpu_gc_after_generation=true and drop_saved_images_from_results=true.
3. Generate 128 txt2img images with Dynamic Prompts, batch size 1, grid disabled, gallery hidden.
4. Confirm files are saved under /workspace/sd/outputs.
5. Confirm generation info/infotext is populated.
6. Record RAM before, peak during, and two minutes after completion.
7. Repeat with 512 images if 128 is stable.
8. Confirm WebUI remains responsive and Infinite Image Browsing can see saved outputs.
```

- [ ] **Step 5: Commit verification notes**

If this repo keeps operational notes, append the RunPod RAM observations to the existing handoff or a new dated note. Otherwise keep the values in the PR description.

```bash
git status --short
```

Expected: only intended source/test/config changes are present.

## Self-Review

- Spec coverage: Option C is covered by Tasks 1 and 2. Option E is covered by Tasks 3, 4, and 6. Existing API/file placeholder behavior is covered by Task 5. Manual RunPod validation from the handoff is covered by Task 7.
- Placeholder scan: No placeholder markers or unspecified "write tests" steps remain.
- Type consistency: The plan consistently uses `devices.cpu_gc(force=False)`, `devices.malloc_trim()`, `processing_output.should_store_saved_image_as_placeholder(...)`, and `processing_output.append_image_result(...)`.
- Review Focus coverage: all five review-focus cases are pinned to Task 1, Task 3, Task 4, Task 5, or Task 7.
