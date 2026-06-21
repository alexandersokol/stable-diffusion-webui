# Console Progress Image Count Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep console progress output visible while reducing terminal redraw overhead and showing the current generated image count at the start of the generation progress line.

**Architecture:** Add a small console progress helper for formatting and option normalization. Track image progress separately from `job_no/job_count`, because those represent sampling jobs/passes rather than generated image totals. Feed the helper into the existing tqdm progress bars with a configurable refresh interval.

**Tech Stack:** Python, tqdm, existing WebUI options/state modules.

---

### Task 1: Console Progress Helpers

**Files:**
- Create: `modules/console_progress.py`
- Test: `test/test_console_progress.py`

- [ ] **Step 1: Add helper functions**

Create `modules/console_progress.py` with:

```python
DEFAULT_MIN_INTERVAL = 0.5


def progress_min_interval(opts) -> float:
    try:
        value = float(getattr(opts, "console_progress_min_interval", DEFAULT_MIN_INTERVAL))
    except (TypeError, ValueError):
        return DEFAULT_MIN_INTERVAL

    return max(0.0, value)


def image_count_label(state) -> str:
    total = int(getattr(state, "console_total_image_count", 0) or 0)
    current = int(getattr(state, "console_current_image_count", 0) or 0)

    if total <= 0:
        return ""

    current = max(0, min(current, total))
    return f"{current}/{total}"


def generation_progress_desc(state, base="Total progress") -> str:
    label = image_count_label(state)
    return f"{label} {base}" if label else base
```

- [ ] **Step 2: Add direct tests**

Create `test/test_console_progress.py` with direct helper coverage for valid intervals, invalid intervals, clamping, and description formatting.

### Task 2: State Tracking

**Files:**
- Modify: `modules/shared_state.py`
- Modify: `modules/processing.py`
- Modify: `modules/img2img.py`

- [ ] **Step 1: Add state fields and methods**

Add `console_current_image_count` and `console_total_image_count`, reset them in `begin()`/`end()`, and add methods to reset, initialize, and advance image count.

- [ ] **Step 2: Initialize normal generation totals**

After `p.setup_prompts()` in `process_images()`, initialize the total to `len(p.all_prompts)` only when no broader total already exists.

- [ ] **Step 3: Update the displayed image count per active batch**

Inside the `for n in range(p.n_iter)` loop, after `p.prompts` is set, update the display count to the upper bound of the active batch so a batch size of 4 shows `4/256` during the first batch.

- [ ] **Step 4: Preserve directory img2img batch totals**

In `img2img.process_batch()`, initialize the total to `len(batch_images) * p.n_iter * p.batch_size` before processing individual files.

### Task 3: Tqdm Integration and Settings

**Files:**
- Modify: `modules/shared_total_tqdm.py`
- Modify: `modules/upscaler_utils.py`
- Modify: `modules/shared_options.py`
- Test: `test/test_shared_total_tqdm.py`
- Test: `test/test_upscaler_utils.py`

- [ ] **Step 1: Use helper in total tqdm**

Set the tqdm description from `console_progress.generation_progress_desc(shared.state)` and pass `mininterval=console_progress.progress_min_interval(shared.opts)`.

- [ ] **Step 2: Refresh description without extra redraws**

Before each total tqdm update, call `set_description_str(desc, refresh=False)` only when the description changes.

- [ ] **Step 3: Use refresh interval for tiled upscaling**

Pass `mininterval=console_progress.progress_min_interval(shared.opts)` into both tiled upscale tqdm calls.

- [ ] **Step 4: Add a setting**

Add `console_progress_min_interval` to the System options with default `0.5`.

- [ ] **Step 5: Add/extend tests**

Add tests that fake tqdm and assert the description starts with `4/256`, the mininterval option is passed through, and tiled upscaling passes mininterval.

### Task 4: Documentation and Verification

**Files:**
- Modify: `IMPROVEMENT_PLAN.md`

- [ ] **Step 1: Mark issue #24 done**

Update issue #24 status from `backlog` to `done`.

- [ ] **Step 2: Verify**

Run compile checks and direct Python test invocations for the new/changed tests.
