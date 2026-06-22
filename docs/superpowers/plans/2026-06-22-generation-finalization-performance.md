# Generation Finalization Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce and expose the post-sampling delay between the final denoise step and the real image file appearing.

**Architecture:** Keep the existing synchronous result contract, but improve the expensive finalization stages in place. Add a PNG compression option used by sample/grid saves, decode latent batches in configurable chunks, and record decode/convert/save timings in generation comments so users can identify the remaining bottleneck.

**Tech Stack:** Python, PyTorch, Pillow, Gradio option definitions, pytest/unittest.

---

### Task 1: Saved PNG Compression Option

**Files:**
- Modify: `modules/shared_options.py`
- Modify: `modules/images.py`
- Test: `test/test_image_save_options.py`

- [ ] **Step 1: Write the failing test**

Create `test/test_image_save_options.py` with a fake options object and a monkeypatched `Image.Image.save` that records keyword arguments:

```python
from types import SimpleNamespace

from PIL import Image

from modules import images


def test_png_save_uses_configured_compress_level(monkeypatch, tmp_path):
    recorded = {}

    def fake_save(self, filename, **kwargs):
        recorded.update(kwargs)
        with open(filename, "wb") as file:
            file.write(b"png")

    monkeypatch.setattr(Image.Image, "save", fake_save)
    monkeypatch.setattr(images, "opts", SimpleNamespace(enable_pnginfo=False, jpeg_quality=80, webp_lossless=False, png_compress_level=1))

    images.save_image_with_geninfo(Image.new("RGB", (1, 1)), "info", str(tmp_path / "image.png"))

    assert recorded["compress_level"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:\stable-diffusion\envs\v1101\python.exe -m pytest test\test_image_save_options.py -q`
Expected: FAIL because `compress_level` is not passed to Pillow.

- [ ] **Step 3: Write minimal implementation**

Add `png_compress_level` under the saving options and pass it to PNG saves:

```python
"png_compress_level": OptionInfo(4, "PNG compression level for saved images", gr.Slider, {"minimum": 0, "maximum": 9, "step": 1}).info("0 saves fastest and largest; 9 saves smallest and slowest"),
```

In `save_image_with_geninfo()`:

```python
image.save(filename, format=image_format, quality=opts.jpeg_quality, pnginfo=pnginfo_data, compress_level=int(opts.png_compress_level))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `D:\stable-diffusion\envs\v1101\python.exe -m pytest test\test_image_save_options.py -q`
Expected: PASS.

### Task 2: Chunked Final VAE Decode

**Files:**
- Modify: `modules/shared_options.py`
- Modify: `modules/processing.py`
- Test: `test/test_processing_decode_chunks.py`

- [ ] **Step 1: Write the failing test**

Create `test/test_processing_decode_chunks.py` with a small fake model and monkeypatched `decode_first_stage`:

```python
from types import SimpleNamespace

import torch

from modules import processing


def test_decode_latent_batch_uses_configured_chunks(monkeypatch):
    calls = []

    def fake_decode_first_stage(model, batch):
        calls.append(batch.shape[0])
        return batch

    monkeypatch.setattr(processing, "decode_first_stage", fake_decode_first_stage)
    monkeypatch.setattr(processing.shared, "opts", SimpleNamespace(final_vae_decode_chunk_size=2, auto_vae_precision=False, auto_vae_precision_bfloat16=False))
    monkeypatch.setattr(processing.devices, "test_for_nans", lambda *_args, **_kwargs: None)

    batch = torch.zeros((5, 3, 2, 2))

    samples = processing.decode_latent_batch(SimpleNamespace(first_stage_model=SimpleNamespace(to=lambda *_args, **_kwargs: None)), batch)

    assert calls == [2, 2, 1]
    assert len(samples) == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:\stable-diffusion\envs\v1101\python.exe -m pytest test\test_processing_decode_chunks.py -q`
Expected: FAIL because the current implementation decodes five single-image calls.

- [ ] **Step 3: Write minimal implementation**

Add `final_vae_decode_chunk_size` as an integer slider. Update `decode_latent_batch()` to read the option, split `batch` by chunk size, call `decode_first_stage()` once per chunk, preserve NaN retry behavior by falling back to per-image retry only for chunks that fail the VAE NaN check, move each decoded sample to `target_device`, and append individual samples to `DecodedSamples`.

- [ ] **Step 4: Run test to verify it passes**

Run: `D:\stable-diffusion\envs\v1101\python.exe -m pytest test\test_processing_decode_chunks.py -q`
Expected: PASS.

### Task 3: Finalization Phase Timing

**Files:**
- Modify: `modules/processing.py`
- Test: `test/test_processing_finalization_timing.py`

- [ ] **Step 1: Write the failing test**

Add focused tests for a helper that only records visible timings when a phase took measurable time:

```python
from types import SimpleNamespace

from modules import processing


def test_add_finalization_timing_comment_records_nonzero_phases():
    p = SimpleNamespace(comments=[])
    timings = {"VAE decode": 0.25, "Image save": 0.5, "Tensor to image": 0.0}

    processing.add_finalization_timing_comment(p, timings)

    assert p.comments == ["Finalization: VAE decode 0.25s, Image save 0.50s"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:\stable-diffusion\envs\v1101\python.exe -m pytest test\test_processing_finalization_timing.py -q`
Expected: FAIL because helper does not exist.

- [ ] **Step 3: Write minimal implementation**

Add a small timing helper in `modules/processing.py`, record `time.perf_counter()` deltas around VAE decode, tensor-to-PIL conversion/postprocess, sample save, and grid save phases, and call the helper before `Processed(...)` is created. Keep comments concise and skip all-zero timings.

- [ ] **Step 4: Run test to verify it passes**

Run: `D:\stable-diffusion\envs\v1101\python.exe -m pytest test\test_processing_finalization_timing.py -q`
Expected: PASS.

### Task 4: Improvement Tracking and Regression Verification

**Files:**
- Modify: `CORE_IMPROVEMENT.md`

- [ ] **Step 1: Mark task #23 done**

Change the `STATUS` cell for ID `23` from `backlog` to `done`.

- [ ] **Step 2: Run focused tests**

Run: `D:\stable-diffusion\envs\v1101\python.exe -m pytest test\test_image_save_options.py test\test_processing_decode_chunks.py test\test_processing_finalization_timing.py -q`
Expected: PASS.

- [ ] **Step 3: Run compile and relevant regression tests**

Run: `D:\stable-diffusion\envs\v1101\python.exe -m py_compile modules\images.py modules\processing.py modules\shared_options.py`
Expected: exit code 0.

Run the existing sanitized regression bundle used for this project and confirm there are no failures.

