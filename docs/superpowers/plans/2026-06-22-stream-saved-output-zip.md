# Stream Saved Output ZIP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stream saved output files into UI-created ZIP archives without reading each file fully into Python memory.

**Architecture:** Keep `modules.ui_common.save_files()` behavior and return values unchanged. Replace the current `open(...).read()` plus `ZipFile.writestr()` loop with `ZipFile.write(full_path, arcname)` so the stdlib ZIP writer streams each file from disk.

**Tech Stack:** Python, stdlib `zipfile`, pytest, existing `modules.ui_common.save_files()` helper.

---

### Task 1: Stream ZIP Entries From Disk

**Files:**
- Create: `test/test_ui_common_save_files.py`
- Modify: `modules/ui_common.py`
- Modify: `CORE_IMPROVEMENT.md`

- [x] **Step 1: Write the failing test**

Create `test/test_ui_common_save_files.py` with a test that:

```python
def test_save_files_streams_zip_entries_with_zipfile_write(tmp_path, monkeypatch):
    ui_common = load_ui_common_for_test()
    saved_files = [tmp_path / "image.png", tmp_path / "image.txt"]
    saved_files[0].write_bytes(b"image-bytes")
    saved_files[1].write_text("parameters", encoding="utf8")
    write_calls = []

    class RecordingZipFile:
        def __init__(self, filename, mode):
            self.filename = filename
            self.mode = mode

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def write(self, filename, arcname=None):
            write_calls.append((filename, arcname))

        def writestr(self, arcname, data):
            raise AssertionError("save_files should stream ZIP entries with ZipFile.write")

    monkeypatch.setattr(zipfile, "ZipFile", RecordingZipFile)

    file_update, html = ui_common.save_files(make_generation_info(), ["gallery-image"], True, -1)

    assert write_calls == [
        (str(saved_files[0]), "image.png"),
        (str(saved_files[1]), "image.txt"),
    ]
    assert file_update["visible"] is True
    assert file_update["value"][0].endswith(".zip")
    assert file_update["value"][1:] == [str(saved_files[0]), str(saved_files[1])]
    assert html == "<p>Saved: image.png</p>"
```

- [x] **Step 2: Run test to verify it fails**

Run:

```powershell
D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_ui_common_save_files.py::test_save_files_streams_zip_entries_with_zipfile_write -q
```

Expected: FAIL because the current implementation calls `ZipFile.writestr()` with bytes read from each saved file.

- [x] **Step 3: Write minimal implementation**

Update `modules/ui_common.py` ZIP creation:

```python
from zipfile import ZipFile
with ZipFile(zip_filepath, "w") as zip_file:
    for fullfn, filename in zip(fullfns, filenames):
        zip_file.write(fullfn, filename)
```

- [x] **Step 4: Mark improvement done**

Update row 18 in `CORE_IMPROVEMENT.md` from `backlog` to `done`.

- [x] **Step 5: Run focused and regression verification**

Run:

```powershell
D:\stable-diffusion\envs\v1101\python.exe -m py_compile modules\ui_common.py test\test_ui_common_save_files.py
D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_ui_common_save_files.py -q
D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_outpainting_mk2.py test/test_sd_models_lookup.py test/test_cache.py test/test_ui_extra_networks_paging.py -q
D:\stable-diffusion\envs\v1101\python.exe -c "import sys, unittest; sys.argv=['unittest']; names=['test.test_sd_hijack_clip_tensor_cache','test.test_prompt_parser_reconstruction','test.test_cfg_denoiser_inputs','test.test_api_image_inputs','test.test_hashes','test.test_sd_models_lookup','test.test_image_hash','test.test_extensions_repo_metadata_cache']; suite=unittest.defaultTestLoader.loadTestsFromNames(names); result=unittest.TextTestRunner(verbosity=2).run(suite); raise SystemExit(0 if result.wasSuccessful() else 1)"
git diff --check
```

Expected: all tests pass; `git diff --check` has no whitespace errors, though Git may report existing CRLF warnings.
