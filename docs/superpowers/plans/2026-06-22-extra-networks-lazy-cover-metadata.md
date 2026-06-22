# Extra Networks Lazy Cover Metadata Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Avoid parsing and materializing embedded cover-image JSON during Extra Networks card rendering.

**Architecture:** Add a cheap raw-string detector for `ssmd_cover_images` in `modules/ui_extra_networks.py`. Use it in `find_embedded_preview()` so card creation can decide whether an embedded cover exists without `json.loads()`, while keeping full JSON parse and base64 decode in `fetch_cover_images()` where image bytes are actually requested.

**Tech Stack:** Python, pytest.

---

### Task 1: Lazy Embedded Cover Detection

**Files:**
- Modify: `test/test_ui_extra_networks_paging.py`
- Modify: `modules/ui_extra_networks.py`
- Modify: `modules/ui_extra_networks_checkpoints.py`
- Modify: `CORE_IMPROVEMENT.md`

- [x] **Step 1: Write the failing tests**

Add tests that patch render-time `json.loads()` to raise while calling `find_embedded_preview()` with a large `ssmd_cover_images` string, and verify it returns the cover endpoint URL without parsing. Add a small endpoint test that confirms `fetch_cover_images()` still parses and returns image bytes on demand.

- [x] **Step 2: Run tests to verify failure**

Run: `D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_ui_extra_networks_paging.py::test_find_embedded_preview_detects_cover_without_json_loads test/test_ui_extra_networks_paging.py::test_fetch_cover_images_decodes_cover_on_demand -q`

Expected: First test FAILS because current `find_embedded_preview()` calls `json.loads()` immediately.

- [x] **Step 3: Implement lazy detector and checkpoint fallback**

Add `metadata_has_embedded_cover(metadata)` or an equivalent helper using raw-string scanning, update `find_embedded_preview()` to use it, and update checkpoint card preview creation to use sidecar preview first then embedded cover fallback.

- [x] **Step 4: Verify focused tests pass**

Run: `D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_ui_extra_networks_paging.py::test_find_embedded_preview_detects_cover_without_json_loads test/test_ui_extra_networks_paging.py::test_fetch_cover_images_decodes_cover_on_demand -q`

Expected: PASS.

- [x] **Step 5: Run regression checks**

Run the Extra Networks paging tests, compile changed files, run prior core regression tests, and run `git diff --check`.
