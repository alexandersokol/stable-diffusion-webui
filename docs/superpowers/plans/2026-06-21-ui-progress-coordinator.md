# UI Progress Coordinator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Centralize browser-side progress polling per task and fan out progress/live-preview updates to all interested UI views.

**Architecture:** `requestProgress()` remains the public entry point used by existing Gradio submit hooks. Internally, it creates a per-task session with one progress polling loop, one optional live-preview loop, and a list of subscriber views. Each subscriber owns its own progress DOM and callbacks, while the task session owns shared network requests and lifecycle cleanup.

**Tech Stack:** Existing vanilla JavaScript, Gradio DOM hooks, existing `/internal/progress` JSON API.

---

### Task 1: Replace One-Caller Guard With Shared Sessions

**Files:**
- Modify: `javascript/progressbar.js`

- [ ] **Step 1: Replace `activeProgressRequests` boolean usage**

Introduce `progressSessions = {}` where each session is keyed by task id and stores:

```javascript
{
    id_task,
    subscribers: [],
    wakeLock: null,
    progressErrors: 0,
    livePreviewErrors: 0,
    progressTimer: null,
    livePreviewTimer: null,
    lastLivePreviewId: 0,
    stopped: false
}
```

- [ ] **Step 2: Add subscriber creation**

Move per-view DOM creation into `createProgressSubscriber(id_task, progressbarContainer, gallery, atEnd, onProgress, inactivityTimeout)`. It creates the `progressDiv`, stores `dateStart`, `wasEverActive`, `livePreview`, `gallery`, callbacks, and a `remove()` method that cleans only that subscriber’s DOM and calls its `atEnd`.

- [ ] **Step 3: Add fan-out update helpers**

Add helpers that render progress text for each subscriber from a shared progress response, update title once per shared response, append live-preview images to each subscriber gallery, and remove subscribers when the task completes, stops being active after being active, or times out inactive.

- [ ] **Step 4: Add shared polling loops**

Make the session start exactly one `pollProgress()` loop and one `pollLivePreview()` loop when a subscriber with a gallery exists. Reuse retry behavior from the old code:

```javascript
setTimeout(pollProgress, Math.min(1000 * session.progressErrors, 5000));
setTimeout(pollLivePreview, Math.min(1000 * session.livePreviewErrors, 5000));
```

The live-preview loop should start later if the first subscriber had no gallery and a later subscriber does.

### Task 2: Preserve Existing Callers And Completion Behavior

**Files:**
- Modify: `javascript/progressbar.js`
- Read-only check: `javascript/ui.js`

- [ ] **Step 1: Keep `requestProgress()` signature unchanged**

Existing callers in `javascript/ui.js`, `javascript/extensions.js`, and `javascript/textualInversion.js` must keep working without edits:

```javascript
function requestProgress(id_task, progressbarContainer, gallery, atEnd, onProgress, inactivityTimeout = 40)
```

- [ ] **Step 2: Remove only duplicate network loops**

When `requestProgress()` is called twice with the same `id_task`, the second call must add a subscriber and must not start another progress loop.

- [ ] **Step 3: Preserve per-view cleanup**

Each subscriber’s `atEnd` callback must still run once, including existing button restoration and local task id cleanup in `javascript/ui.js`.

### Task 3: Update Plan Status And Verify

**Files:**
- Modify: `IMPROVEMENT_PLAN.md`

- [ ] **Step 1: Mark issue 20 done**

Change the status for issue 20 from `backlog` to `done`.

- [ ] **Step 2: Run syntax and source checks**

Run:

```powershell
D:\stable-diffusion\envs\v1101\python.exe -m compileall -f modules\progress.py test\test_progress.py
D:\stable-diffusion\envs\v1101\python.exe -c "from pathlib import Path; p=Path('javascript/progressbar.js').read_text(); assert 'progressSessions' in p; assert 'activeProgressRequests' not in p; assert p.count('request(\"./internal/progress\"') == 2; print('progress coordinator source checks passed')"
D:\stable-diffusion\envs\v1101\python.exe -c "from test import test_progress as t; t.test_progress_task_lifecycle_is_visible_through_locked_snapshots(); t.test_progress_active_task_without_start_time_returns_valid_response(); t.test_record_results_keeps_only_limit(); t.test_record_results_stores_lightweight_gallery_snapshot(); t.test_restore_progress_waits_until_task_finishes(); t.test_resize_live_preview_image_caps_largest_side(); print('progress tests passed')"
git diff --check
```

Expected: all commands succeed. CRLF warnings from `git diff --check` are acceptable if they are only line-ending notices.
