# Outpainting Matched Noise Vectorization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Vectorize the hot NumPy math inside Outpainting mk2 matched-noise generation without changing generated pixels.

**Architecture:** Keep the public `get_matched_noise(_np_src_image, np_mask_rgb, noise_q=1, color_variation=0.05)` API unchanged. Replace per-channel FFT/window/mask loops with NumPy axis-aware FFT calls and broadcasting, while a focused test compares the vectorized output against the legacy loop implementation.

**Tech Stack:** Python, NumPy, scikit-image, pytest, PIL import stubs for loading the script module without launching the web UI stack.

---

### Task 1: Matched Noise Vectorization

**Files:**
- Create: `test/test_outpainting_mk2.py`
- Modify: `scripts/outpainting_mk_2.py`
- Modify: `CORE_IMPROVEMENT.md`

- [x] **Step 1: Write the failing test**

Create `test/test_outpainting_mk2.py` with:

```python
def test_get_matched_noise_matches_legacy_output_and_uses_stacked_fft(monkeypatch):
    outpainting = load_outpainting_module_for_test()
    src, mask = make_test_image_and_mask()
    expected = legacy_get_matched_noise(src.copy(), mask.copy(), noise_q=1.25, color_variation=0.35)
    original_fft2 = outpainting.np.fft.fft2
    original_ifft2 = outpainting.np.fft.ifft2
    fft2_shapes = []
    ifft2_shapes = []

    def recording_fft2(data, *args, **kwargs):
        fft2_shapes.append(data.shape)
        return original_fft2(data, *args, **kwargs)

    def recording_ifft2(data, *args, **kwargs):
        ifft2_shapes.append(data.shape)
        return original_ifft2(data, *args, **kwargs)

    monkeypatch.setattr(outpainting.np.fft, "fft2", recording_fft2)
    monkeypatch.setattr(outpainting.np.fft, "ifft2", recording_ifft2)

    actual = outpainting.get_matched_noise(src.copy(), mask.copy(), noise_q=1.25, color_variation=0.35)

    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert fft2_shapes == [(8, 10, 3), (8, 10, 3), (8, 10, 3)]
    assert ifft2_shapes == [(8, 10, 3), (8, 10, 3)]
```

- [x] **Step 2: Run test to verify it fails**

Run:

```powershell
D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_outpainting_mk2.py::test_get_matched_noise_matches_legacy_output_and_uses_stacked_fft -q
```

Expected: FAIL because the current implementation records nine 2D `fft2` calls and six 2D `ifft2` calls instead of three and two stacked RGB calls.

- [x] **Step 3: Write minimal implementation**

Update `scripts/outpainting_mk_2.py`:

```python
def _fft2(data):
    if data.ndim > 2:
        return np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(data, axes=(0, 1)), axes=(0, 1), norm="ortho"), axes=(0, 1))

    return np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(data), norm="ortho"))

def _ifft2(data):
    if data.ndim > 2:
        return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(data, axes=(0, 1)), axes=(0, 1), norm="ortho"), axes=(0, 1))

    return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(data), norm="ortho"))
```

Vectorize `_get_gaussian_window()`, `_get_masked_window_rgb()`, noise grayscale blending, and `noise_fft *= noise_window[:, :, None]`.

- [x] **Step 4: Mark improvement done**

Update row 17 in `CORE_IMPROVEMENT.md` from `backlog` to `done`.

- [x] **Step 5: Run focused and regression verification**

Run:

```powershell
D:\stable-diffusion\envs\v1101\python.exe -m py_compile scripts\outpainting_mk_2.py test\test_outpainting_mk2.py
D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_outpainting_mk2.py -q
D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_sd_models_lookup.py test/test_cache.py test/test_ui_extra_networks_paging.py -q
D:\stable-diffusion\envs\v1101\python.exe -c "import sys, unittest; sys.argv=['unittest']; names=['test.test_sd_hijack_clip_tensor_cache','test.test_prompt_parser_reconstruction','test.test_cfg_denoiser_inputs','test.test_api_image_inputs','test.test_hashes','test.test_sd_models_lookup','test.test_image_hash','test.test_extensions_repo_metadata_cache']; suite=unittest.defaultTestLoader.loadTestsFromNames(names); result=unittest.TextTestRunner(verbosity=2).run(suite); raise SystemExit(0 if result.wasSuccessful() else 1)"
git diff --check
```

Expected: all tests pass; `git diff --check` has no whitespace errors, though Git may report existing CRLF warnings.
