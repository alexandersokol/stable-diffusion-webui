# Console Generation Stage Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show image position and current generation stage in the console progress bar.

**Architecture:** Extend the existing console progress state instead of printing extra lines. `shared_state.State` owns the current console image count and stage, `console_progress` formats the bracketed labels and tqdm bar format, `shared_total_tqdm.TotalTQDM` applies that format, and `processing.py` updates stages along the generation pipeline.

**Tech Stack:** Python, tqdm, pytest/unittest.

---

### Task 1: Progress Formatting

**Files:**
- Modify: `modules/console_progress.py`
- Test: `test/test_console_progress.py`

- [ ] **Step 1: Write failing tests**

Add tests expecting:

```python
state = SimpleNamespace(console_current_image_count=1, console_total_image_count=100, console_generation_stage="DONE")
assert console_progress.image_count_label(state) == "[1/100]"
assert console_progress.stage_label(state) == "[DONE]"
assert console_progress.generation_progress_desc(state) == "[1/100]"
assert console_progress.generation_progress_bar_format(state) == "{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] [DONE]"
```

- [ ] **Step 2: Run test and observe failure**

Run: `D:\stable-diffusion\envs\v1101\python.exe -m pytest test\test_console_progress.py -q`
Expected: FAIL because current labels are unbracketed and stage/bar format helpers do not exist.

- [ ] **Step 3: Implement formatting helpers**

Update `image_count_label()`, add `stage_label()`, add `generation_progress_bar_format()`, and keep `progress_min_interval()` unchanged.

### Task 2: tqdm Wiring

**Files:**
- Modify: `modules/shared_total_tqdm.py`
- Test: `test/test_shared_total_tqdm.py`

- [ ] **Step 1: Write failing tests**

Update FakeTqdm expectations so construction includes `bar_format`, initial `desc` is `[4/256]`, and stage changes refresh the description/bar format without emitting a separate progress line.

- [ ] **Step 2: Run test and observe failure**

Run: `D:\stable-diffusion\envs\v1101\python.exe -m pytest test\test_shared_total_tqdm.py -q`
Expected: FAIL because `bar_format` is not supplied and stage updates are not tracked.

- [ ] **Step 3: Implement tqdm wiring**

Pass `bar_format=console_progress.generation_progress_bar_format(shared.state)` at reset, update description and bar format on each update, and call `refresh()` when the bar format changes so the stage appears immediately.

### Task 3: Generation Stage State

**Files:**
- Modify: `modules/shared_state.py`
- Modify: `modules/processing.py`
- Test: `test/test_shared_state.py`

- [ ] **Step 1: Write failing tests**

Add state tests for reset/setup/stage mutation:

```python
state = State()
state.set_console_generation_stage("SAMPLING")
assert state.console_generation_stage == "SAMPLING"
state.reset_console_image_progress()
assert state.console_generation_stage == ""
```

- [ ] **Step 2: Run test and observe failure**

Run: `D:\stable-diffusion\envs\v1101\python.exe -m pytest test\test_shared_state.py -q`
Expected: FAIL because stage state does not exist.

- [ ] **Step 3: Implement state and processing updates**

Add stage state to `State`, reset it with console progress, and update `processing.py` with these stages:

`PREP` before init/conds, `SAMPLING` before sampler call, `VAE` during final decode, `POST` during tensor/PIL/postprocess work, `SAVING` during sample saves, `GRID` while building/saving grids, `DONE` after finalization and before return.

### Task 4: Verification

**Files:**
- No additional files.

- [ ] **Step 1: Run focused tests**

Run: `D:\stable-diffusion\envs\v1101\python.exe -m pytest test\test_console_progress.py test\test_shared_total_tqdm.py test\test_shared_state.py -q`

- [ ] **Step 2: Compile changed files**

Run: `D:\stable-diffusion\envs\v1101\python.exe -m py_compile modules\console_progress.py modules\shared_total_tqdm.py modules\shared_state.py modules\processing.py`

- [ ] **Step 3: Run relevant regression bundle**

Run existing progress and processing-related tests plus the sanitized unittest bundle.

