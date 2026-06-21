# Output Storage Log Rotation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent `log.csv` from growing forever and expose output folder usage visibility without deleting generated image metadata sidecars.

**Architecture:** Add a focused output storage helper for `log.csv` rotation and output directory scanning. Wire rotation into the gallery Save action immediately before opening `log.csv` for append. Add settings and an Actions tab report button so users can see image, `.txt`, and log storage usage.

**Tech Stack:** Python standard library filesystem APIs, Gradio settings/actions, existing WebUI option patterns.

---

### Task 1: Output Storage Helper

**Files:**
- Create: `modules/output_storage.py`
- Test: `test/test_output_storage.py`

- [ ] **Step 1: Add log rotation**

Create `modules/output_storage.py` with helpers to normalize MB/count settings and rotate `log.csv` when it exceeds the configured size. Rotation renames `log.csv` to `log.csv.1`, shifts older numbered backups upward, and removes backups beyond the configured count.

- [ ] **Step 2: Add output usage reporting**

Add scanning helpers that recursively summarize configured output folders, counting total bytes/files, image bytes/files, `.txt` sidecar bytes/files, and `log.csv*` bytes/files. Include the largest output folders in the report.

- [ ] **Step 3: Add direct tests**

Create tests for rotation, disabled rotation, backup count pruning, output folder summaries, and HTML-safe reporting.

### Task 2: Wire Runtime and UI

**Files:**
- Modify: `modules/ui_common.py`
- Modify: `modules/ui_settings.py`
- Modify: `modules/shared_options.py`

- [ ] **Step 1: Rotate before append**

Call `output_storage.rotate_log_file_if_needed(logfile_path)` before `update_logfile()` and before appending to `log.csv`.

- [ ] **Step 2: Add settings**

Add `save_log_max_size_mb` and `save_log_backup_count` in the Saving images options section.

- [ ] **Step 3: Add report action**

Add a Settings → Actions button that calls `output_storage.output_storage_report`.

### Task 3: Plan Status and Verification

**Files:**
- Modify: `IMPROVEMENT_PLAN.md`

- [ ] **Step 1: Mark issue #25 done**

Update the issue #25 row to `done`.

- [ ] **Step 2: Verify**

Run compile checks, direct tests for output storage, and `git diff --check`.
