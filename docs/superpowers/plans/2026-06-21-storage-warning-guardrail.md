# Storage Warning Guardrail Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Warn before txt2img/img2img jobs that are likely to write a large amount of image data to disk.

**Architecture:** Add a small Python estimator for testable storage math and a matching browser-side pre-submit estimator for immediate confirmation before generation starts. The JavaScript reads current UI dimensions/batch values and existing `opts` settings, then aborts the submit if the user cancels the warning.

**Tech Stack:** Existing vanilla JavaScript submit hooks, existing options JSON, Python unit-style direct tests.

---

### Task 1: Add Testable Storage Estimator

**Files:**
- Create: `modules/storage_estimation.py`
- Create: `test/test_storage_estimation.py`

- [ ] **Step 1: Add storage math helpers**

Create dataclasses for generation context and saving options. Estimate bytes from width, height, image count, grid count, metadata sidecars, and heavy copy multipliers.

- [ ] **Step 2: Add formatting helper**

Add `format_bytes()` for user-facing MB/GB display.

- [ ] **Step 3: Add direct tests**

Test base sample estimate, grid estimate, sidecar estimate, and img2img/init/mask multipliers.

### Task 2: Add User Options

**Files:**
- Modify: `modules/shared_options.py`

- [ ] **Step 1: Add warning settings**

Add:
- `storage_warning_enabled`, default `True`
- `storage_warning_threshold_mb`, default `1024`

Place them in the existing Saving images/grids section.

### Task 3: Add Pre-Submit Browser Warning

**Files:**
- Modify: `javascript/ui.js`

- [ ] **Step 1: Read UI values**

Add helpers to read numeric Gradio inputs by element id, estimate txt2img final dimensions when hires fix is enabled, and estimate image-like output bytes.

- [ ] **Step 2: Build warning text**

Warn only when the estimated bytes exceed `opts.storage_warning_threshold_mb`.

- [ ] **Step 3: Abort before task creation**

Call the warning helper at the start of `submit()` and `submit_img2img()`, before hiding buttons or creating task ids. If the user cancels, throw a small cancellation error so Gradio does not submit the backend job.

### Task 4: Mark Done And Verify

**Files:**
- Modify: `IMPROVEMENT_PLAN.md`

- [ ] **Step 1: Mark issue 21 done**

Change issue 21 status from `backlog` to `done`.

- [ ] **Step 2: Run checks**

Run:

```powershell
D:\stable-diffusion\envs\v1101\python.exe -m compileall -f modules\storage_estimation.py test\test_storage_estimation.py
D:\stable-diffusion\envs\v1101\python.exe -c "from test import test_storage_estimation as t; t.test_estimates_saved_samples_and_txt_sidecars(); t.test_estimates_grid_as_one_large_image(); t.test_estimates_img2img_extra_outputs(); t.test_format_bytes_uses_mb_and_gb(); print('storage estimation tests passed')"
node --check javascript\ui.js
git diff --check
```

Expected: all checks exit 0; CRLF warnings are acceptable if they are only line-ending notices.
