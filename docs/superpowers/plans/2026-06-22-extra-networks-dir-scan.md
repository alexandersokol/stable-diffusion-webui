# Extra Networks Directory Scan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Extra Networks directory button rendering stream directory traversal without materializing the whole tree or calling `os.listdir()` for every subdirectory.

**Architecture:** Add a small `os.scandir()` recursive helper in `modules/ui_extra_networks.py` that yields subdirectories in natural-sort order with a precomputed `has_children` flag. Use that helper from `ExtraNetworksPage.create_dirs_view_html()` while preserving existing hidden-directory filtering and button path formatting.

**Tech Stack:** Python, pytest.

---

### Task 1: Stream Directory Button Scan

**Files:**
- Modify: `test/test_ui_extra_networks_paging.py`
- Modify: `modules/ui_extra_networks.py`
- Modify: `CORE_IMPROVEMENT.md`

- [x] **Step 1: Write the failing test**

Add a test that creates empty, non-empty, and hidden subdirectories, patches `modules.ui_extra_networks.os.listdir` to raise, calls `create_dirs_view_html()`, and verifies the HTML still contains the visible directory buttons with the correct trailing separator behavior while excluding hidden directories.

- [x] **Step 2: Run test to verify it fails**

Run: `D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_ui_extra_networks_paging.py::test_create_dirs_view_html_uses_scandir_without_per_directory_listdir -q`

Expected: FAIL because the current `create_dirs_view_html()` calls `os.listdir()` for each discovered directory.

- [x] **Step 3: Implement streaming scan**

Add a module-level `walk_preview_directories(root)` helper using `os.scandir()`. It should sort child directories one level at a time, yield `(path, has_children)` for each directory, and recurse into child directories. Replace the current `sorted(os.walk(...))` plus `os.listdir()` loop in `create_dirs_view_html()`.

- [x] **Step 4: Verify focused tests pass**

Run: `D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_ui_extra_networks_paging.py::test_create_dirs_view_html_uses_scandir_without_per_directory_listdir -q`

Expected: PASS.

- [x] **Step 5: Run regression checks**

Run the Extra Networks paging tests, compile changed files, run prior core regression tests, and run `git diff --check`.
