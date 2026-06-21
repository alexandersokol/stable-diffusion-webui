# API File Response Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in file-reference response mode for txt2img/img2img API generation so large batches do not have to duplicate every output image as base64 in memory.

**Architecture:** Keep the existing base64 response as the default for compatibility. Add a request field that selects `base64` or `file`; in `file` mode, force returned outputs to be saved and return their saved paths in a new response field.

**Tech Stack:** Python, FastAPI/Pydantic models, existing Stable Diffusion WebUI processing and image-save helpers.

---

### Task 1: API Model Fields

**Files:**
- Modify: `modules/api/models.py`

- [x] Add `image_return_mode` to txt2img and img2img request models with default `"base64"`.
- [x] Add `image_paths` to txt2img and img2img response models.

### Task 2: API Response Helpers

**Files:**
- Modify: `modules/api/api.py`

- [x] Add a mode normalizer that accepts only `"base64"` and `"file"`.
- [x] Add a helper that returns either base64 images or saved file paths.
- [x] In file mode, do not call `encode_pil_to_base64`.

### Task 3: Wire Generation Endpoints

**Files:**
- Modify: `modules/api/api.py`

- [x] Read `image_return_mode` from txt2img/img2img requests.
- [x] Force sample/grid saving when file mode is requested.
- [x] Return `image_paths` in response objects.

### Task 4: Tests And Plan Status

**Files:**
- Create: `test/test_api_image_response.py`
- Modify: `IMPROVEMENT_PLAN.md`

- [x] Add direct tests for mode validation, base64 mode, file mode, and `send_images=false`.
- [x] Mark issue #26 as `done`.
- [x] Run compile and direct test verification.
