# Auxiliary Temp Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Settings action that reports and cleans known safe auxiliary temporary files.

**Architecture:** Create a small `modules/maintenance.py` module with explicit allowlisted cleanup targets. It deletes Gradio theme cache JSON files and moved corrupt `tmp/config.json` / `tmp/cache.json` files, then invokes the existing WebUI-owned temp-file cleanup wrapper.

**Tech Stack:** Existing Python settings UI, filesystem helpers, direct temp-dir tests.

---

### Task 1: Add Maintenance Cleanup Helpers

**Files:**
- Create: `modules/maintenance.py`
- Modify: `modules/ui_tempdir.py`
- Create: `test/test_maintenance.py`

- [ ] **Step 1: Add safe target discovery**

Add helpers for `tmp/gradio_themes/*.json` and `tmp/config.json` / `tmp/cache.json`.

- [ ] **Step 2: Add delete/report helper**

Delete only discovered allowlisted files and return counts/sizes.

- [ ] **Step 3: Expose temp-dir cleanup report**

Add `cleanup_tmpdir_report()` in `modules/ui_tempdir.py` so maintenance can include WebUI-owned temp files without changing ownership rules.

### Task 2: Wire Settings Action

**Files:**
- Modify: `modules/ui_settings.py`

- [ ] **Step 1: Import maintenance**

Add `maintenance` to the existing modules import.

- [ ] **Step 2: Add button**

Add `Cleanup auxiliary temp files` beside existing cleanup buttons.

- [ ] **Step 3: Wire button**

Connect it to `maintenance.cleanup_auxiliary_temp_files_report`.

### Task 3: Mark Done And Verify

**Files:**
- Modify: `IMPROVEMENT_PLAN.md`

- [ ] **Step 1: Mark issue 23 done**

Change issue 23 status from `backlog` to `done`.

- [ ] **Step 2: Run checks**

Run:

```powershell
D:\stable-diffusion\envs\v1101\python.exe -m compileall -f modules\maintenance.py modules\ui_tempdir.py modules\ui_settings.py test\test_maintenance.py
D:\stable-diffusion\envs\v1101\python.exe -c "from test import test_maintenance as t; t.test_cleanup_removes_gradio_theme_json_only(); t.test_cleanup_removes_moved_config_and_cache_only(); t.test_cleanup_report_includes_tempdir_result(); print('maintenance tests passed')"
git diff --check
```

Expected: all checks exit 0; CRLF warnings are acceptable if they are only line-ending notices.
