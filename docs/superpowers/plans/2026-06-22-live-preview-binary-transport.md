# Live Preview Binary Transport Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move internal live-preview image transport from base64 JSON payloads to binary blob responses.

**Architecture:** Keep `/internal/progress` as the JSON status endpoint and use it only to report `id_live_preview`. Add `/internal/live-preview` for binary image bytes, cached by the existing preview cache key. Update the browser polling code to fetch preview blobs when the preview id changes, display object URLs, and revoke old URLs.

**Tech Stack:** Python, FastAPI/Starlette `Response`, PIL, pytest, browser `XMLHttpRequest` blobs and object URLs.

---

### Task 1: Backend Binary Preview Contract

**Files:**
- Modify: `test/test_progress.py`
- Modify: `modules/progress.py`

- [x] **Step 1: Write failing backend tests**

Add tests that assert:

```python
preview_bytes, media_type = progress._encode_live_preview(image, 1)
assert isinstance(preview_bytes, bytes)
assert media_type == "image/png"
assert preview_bytes == second_preview_bytes
```

Add a progress response test:

```python
response = progress.progressapi(progress.ProgressRequest(id_task=task_id, id_live_preview=-1, live_preview=True))
assert response.live_preview is None
assert response.id_live_preview == shared.state.id_live_preview
```

Add a binary endpoint test:

```python
response = progress.live_preview_api(progress.LivePreviewRequest(id_task=task_id, id_live_preview=shared.state.id_live_preview))
assert response.media_type == "image/png"
assert response.body.startswith(b"\x89PNG")
```

- [x] **Step 2: Run tests to verify they fail**

Run:

```powershell
D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_progress.py::test_encode_live_preview_returns_cached_image_bytes test/test_progress.py::test_progress_response_reports_preview_id_without_json_payload test/test_progress_live_preview_endpoint_returns_current_preview_bytes -q
```

Expected: FAIL because `_encode_live_preview()` currently returns a data URI string and `LivePreviewRequest`/`live_preview_api()` do not exist yet.

- [x] **Step 3: Implement backend**

Update `modules/progress.py`:

```python
class LivePreviewRequest(BaseModel):
    id_task: Optional[str] = Field(default=None, title="Task ID")
    id_live_preview: int = Field(default=-1, title="Live preview image ID")

def _encode_live_preview(image, id_live_preview):
    ...
    preview = (buffered.getvalue(), f"image/{image_format}")
    ...
    return preview

def live_preview_api(req: LivePreviewRequest):
    ...
    return Response(content=preview_bytes, media_type=media_type)
```

Register `/internal/live-preview` in `setup_progress_api()`. Keep `/internal/progress` from putting preview bytes/base64 into JSON.

### Task 2: Frontend Blob Fetch

**Files:**
- Modify: `javascript/progressbar.js`
- Modify: `CORE_IMPROVEMENT.md`

- [x] **Step 1: Implement blob fetch helper**

Add a request helper that posts JSON and returns `xhr.response` with `responseType = "blob"`.

- [x] **Step 2: Update live preview polling**

After `/internal/progress` reports a changed `id_live_preview`, call `/internal/live-preview` and pass `URL.createObjectURL(blob)` to `updateLivePreviewSubscriber()`. Track and revoke object URLs in `updateLivePreviewSubscriber()` after image load/error and when preview elements are trimmed.

- [x] **Step 3: Mark improvement done**

Update row 20 in `CORE_IMPROVEMENT.md` from `backlog` to `done`.

- [x] **Step 4: Run verification**

Run:

```powershell
D:\stable-diffusion\envs\v1101\python.exe -m py_compile modules\progress.py test\test_progress.py
D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_progress.py -q
D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_interrogate_cache.py test/test_ui_common_save_files.py test/test_outpainting_mk2.py test/test_sd_models_lookup.py test/test_cache.py test/test_ui_extra_networks_paging.py -q
D:\stable-diffusion\envs\v1101\python.exe -c "import sys, unittest; sys.argv=['unittest']; names=['test.test_sd_hijack_clip_tensor_cache','test.test_prompt_parser_reconstruction','test.test_cfg_denoiser_inputs','test.test_api_image_inputs','test.test_hashes','test.test_sd_models_lookup','test.test_image_hash','test.test_extensions_repo_metadata_cache']; suite=unittest.defaultTestLoader.loadTestsFromNames(names); result=unittest.TextTestRunner(verbosity=2).run(suite); raise SystemExit(0 if result.wasSuccessful() else 1)"
git diff --check
```

Expected: all tests pass; `git diff --check` has no whitespace errors, though Git may report existing CRLF warnings.
