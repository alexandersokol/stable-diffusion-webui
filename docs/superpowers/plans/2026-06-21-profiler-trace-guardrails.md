# Profiler Trace Guardrails Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bound Torch profiler trace storage growth and provide a manual cleanup action.

**Architecture:** Keep the latest profiler output at the configured `profiling_filename` so existing UI links keep working. Before exporting a new trace, rotate any existing trace to a timestamped sibling file, then prune profiler traces by configured maximum file count and total size.

**Tech Stack:** Existing Python profiler wrapper, Gradio Settings Actions, direct Python tests with temporary directories.

---

### Task 1: Add Profiling Trace File Helpers

**Files:**
- Modify: `modules/profiling.py`
- Create: `test/test_profiling.py`

- [ ] **Step 1: Add path and discovery helpers**

Add helpers to resolve the configured trace file, discover the latest file plus timestamped rotated siblings, and compute total size.

- [ ] **Step 2: Add rotation and pruning helpers**

Before export, rotate an existing trace to a timestamped sibling. After export, remove oldest discovered profiler traces until both configured count and total-size budgets are satisfied.

- [ ] **Step 3: Add cleanup report helper**

Expose `cleanup_profile_traces_report()` returning a short string for the Settings UI.

### Task 2: Wire Export And Settings

**Files:**
- Modify: `modules/profiling.py`
- Modify: `modules/shared_options.py`
- Modify: `modules/ui_settings.py`

- [ ] **Step 1: Use guarded export**

Replace direct `export_chrome_trace(shared.opts.profiling_filename)` with a guarded export helper.

- [ ] **Step 2: Add settings**

Add `profiling_max_trace_files` and `profiling_max_total_size_mb` to the Profiler settings section, and update the profiler explanation.

- [ ] **Step 3: Add Settings action**

Add a “Cleanup profiling traces” button in Settings > Actions wired to `profiling.cleanup_profile_traces_report`.

### Task 3: Mark Done And Verify

**Files:**
- Modify: `IMPROVEMENT_PLAN.md`

- [ ] **Step 1: Mark issue 22 done**

Change issue 22 status from `backlog` to `done`.

- [ ] **Step 2: Run checks**

Run:

```powershell
D:\stable-diffusion\envs\v1101\python.exe -m compileall -f modules\profiling.py modules\shared_options.py modules\ui_settings.py test\test_profiling.py
D:\stable-diffusion\envs\v1101\python.exe -c "from test import test_profiling as t; t.test_rotate_existing_trace_keeps_configured_latest_path(); t.test_cleanup_profile_traces_prunes_by_count(); t.test_cleanup_profile_traces_prunes_by_total_size(); t.test_cleanup_profile_traces_report_returns_reclaimed_size(); print('profiling tests passed')"
git diff --check
```

Expected: all checks exit 0; CRLF warnings are acceptable if they are only line-ending notices.
