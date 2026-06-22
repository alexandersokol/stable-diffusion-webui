# Shared Generation Progress Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep txt2img/img2img progress, live preview, and Generate/Interrupt/Skip controls synced across all browsers connected to the same WebUI server.

**Architecture:** The backend `/internal/progress` response remains the source of truth for the currently active task and adds interruption flags needed by clients. The frontend periodically probes the server for the active txt2img/img2img task, attaches progress/live preview for clients that did not start the task, and derives button visibility from the shared progress response.

**Tech Stack:** Python, Pydantic/FastAPI response models, Gradio WebUI JavaScript, pytest, Node.js for focused JavaScript helper tests.

---

### Task 1: Backend Progress Flags

**Files:**
- Modify: `modules/progress.py`
- Test: `test/test_progress.py`

- [ ] **Step 1: Write failing backend test**

Add a test that starts a txt2img task, sets shared interruption flags, calls `progressapi`, and asserts the response exposes `interrupted` and `stopping_generation`.

- [ ] **Step 2: Run backend test to verify failure**

Run: `D:\stable-diffusion\envs\v1101\python.exe -m pytest test\test_progress.py::test_progress_response_reports_interruption_flags -q`

Expected: FAIL because the response model does not expose the new flags yet.

- [ ] **Step 3: Add response fields**

Add `interrupted`, `skipped`, and `stopping_generation` boolean fields to `ProgressResponse`, then populate them from `shared.state` for active responses and use defaults for inactive responses.

- [ ] **Step 4: Run backend test to verify pass**

Run: `D:\stable-diffusion\envs\v1101\python.exe -m pytest test\test_progress.py::test_progress_response_reports_interruption_flags -q`

Expected: PASS.

### Task 2: Frontend Shared Task Attachment

**Files:**
- Modify: `javascript/ui.js`
- Test: `test/test_ui_progress_sync_js.py`

- [ ] **Step 1: Write failing JavaScript helper tests**

Create pytest tests that run Node against `javascript/progressbar.js` and `javascript/ui.js`, checking:
- active txt2img/img2img responses should attach when no matching progress session exists;
- extras/inactive responses should not attach;
- active interruption flags should map to the Interrupting button state.

- [ ] **Step 2: Run JavaScript helper tests to verify failure**

Run: `D:\stable-diffusion\envs\v1101\python.exe -m pytest test\test_ui_progress_sync_js.py -q`

Expected: FAIL because the helper functions do not exist yet.

- [ ] **Step 3: Implement shared progress helpers**

In `javascript/ui.js`, add small helpers for deciding whether a server task is attachable and which button state an active progress response should show.

- [ ] **Step 4: Add continuous server monitor**

Replace the one-shot `attachCurrentServerProgress()` behavior with a monitor that polls `/internal/progress` without a task id at a modest interval. When it sees an active txt2img/img2img task that is not already attached, call `attachProgress()` for that tab.

- [ ] **Step 5: Sync button state during progress polling**

Pass an `onProgress` callback to `requestProgress()` from `attachProgress()` so all attached clients update Generate/Interrupt/Skip/Interrupting visibility from the shared progress flags.

- [ ] **Step 6: Run JavaScript helper tests to verify pass**

Run: `D:\stable-diffusion\envs\v1101\python.exe -m pytest test\test_ui_progress_sync_js.py -q`

Expected: PASS.

### Task 3: Regression Verification

**Files:**
- Verify only.

- [ ] **Step 1: Run focused tests**

Run: `D:\stable-diffusion\envs\v1101\python.exe -m pytest test\test_progress.py test\test_ui_progress_sync_js.py -q`

Expected: PASS.

- [ ] **Step 2: Compile touched Python**

Run: `D:\stable-diffusion\envs\v1101\python.exe -m py_compile modules\progress.py`

Expected: exit code 0.

- [ ] **Step 3: Run broader related tests**

Run: `D:\stable-diffusion\envs\v1101\python.exe -m pytest test\test_progress.py test\test_progress_results.py test\test_txt2img.py test\test_img2img.py -q`

Expected: PASS or existing environment-dependent API tests are reported explicitly.
