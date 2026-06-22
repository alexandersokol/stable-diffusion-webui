# Textual Inversion Image Embedding Vectorization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Vectorize textual inversion image-embedding helper hot paths while preserving embedded PNG compatibility and caption overlay appearance.

**Architecture:** Keep the public helper behavior unchanged. Add a NumPy-backed `lcg_bytes(count)` helper for deterministic pseudo-random bytes, use it in `xor_block()`, and build caption overlay alpha gradients as a NumPy array instead of writing one pixel at a time.

**Tech Stack:** Python, NumPy, PIL, PyTorch, pytest.

---

### Task 1: Vectorized LCG Bytes And Caption Gradient

**Files:**
- Create: `test/test_textual_inversion_image_embedding.py`
- Modify: `modules/textual_inversion/image_embedding.py`
- Modify: `CORE_IMPROVEMENT.md`

- [x] **Step 1: Write failing tests**

Add tests that assert:

```python
assert image_embedding.lcg_bytes(100).tolist() == reference_random
assert np.array_equal(image_embedding.xor_block(block), legacy_xor_block(block))
```

Add a caption overlay test that patches `Image.Image.putpixel` to raise and verifies the resulting alpha channel matches the legacy gradient formula outside the text rows:

```python
monkeypatch.setattr(Image.Image, "putpixel", fail_putpixel)
overlay = image_embedding.caption_image_overlay(image, "", "", "", "")
assert np.array_equal(np.asarray(overlay.getchannel("A"))[:, 0], expected_alpha)
```

- [x] **Step 2: Run tests to verify they fail**

Run:

```powershell
D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_textual_inversion_image_embedding.py -q
```

Expected: FAIL because `lcg_bytes()` does not exist and `caption_image_overlay()` still calls `putpixel()`.

- [x] **Step 3: Implement vectorized helpers**

Update `modules/textual_inversion/image_embedding.py`:

```python
def lcg_bytes(count, m=2**32, a=1664525, c=1013904223, seed=0):
    return np.fromiter(lcg(m=m, a=a, c=c, seed=seed), dtype=np.uint32, count=count).astype(np.uint8)

def xor_block(block):
    randblock = lcg_bytes(np.prod(block.shape)).reshape(block.shape)
    return np.bitwise_xor(block.astype(np.uint8), randblock & 0x0F)
```

Build the caption alpha gradient with vectorized `np.arange`, `np.cos`, and `Image.fromarray()`.

- [x] **Step 4: Mark improvement done**

Update row 21 in `CORE_IMPROVEMENT.md` from `backlog` to `done`.

- [x] **Step 5: Run focused and regression verification**

Run:

```powershell
D:\stable-diffusion\envs\v1101\python.exe -m py_compile modules\textual_inversion\image_embedding.py test\test_textual_inversion_image_embedding.py
D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_textual_inversion_image_embedding.py -q
D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_progress.py test/test_interrogate_cache.py test/test_ui_common_save_files.py test/test_outpainting_mk2.py test/test_sd_models_lookup.py test/test_cache.py test/test_ui_extra_networks_paging.py -q
D:\stable-diffusion\envs\v1101\python.exe -c "import sys, unittest; sys.argv=['unittest']; names=['test.test_sd_hijack_clip_tensor_cache','test.test_prompt_parser_reconstruction','test.test_cfg_denoiser_inputs','test.test_api_image_inputs','test.test_hashes','test.test_sd_models_lookup','test.test_image_hash','test.test_extensions_repo_metadata_cache']; suite=unittest.defaultTestLoader.loadTestsFromNames(names); result=unittest.TextTestRunner(verbosity=2).run(suite); raise SystemExit(0 if result.wasSuccessful() else 1)"
git diff --check
```

Expected: all tests pass; `git diff --check` has no whitespace errors, though Git may report existing CRLF warnings.
