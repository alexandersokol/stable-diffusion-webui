# CLIP Interrogator Batched Ranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Batch CLIP interrogator label ranking math so multi-row image features use one matrix multiply and one softmax.

**Architecture:** Keep `InterrogateModels.rank()` input/output behavior unchanged. Replace the Python loop over `image_features` rows with batched `image_features @ text_features.T`, softmax over labels, and mean over image rows.

**Tech Stack:** Python, PyTorch, pytest/unittest-compatible existing interrogator tests.

---

### Task 1: Batched Rank Similarity

**Files:**
- Modify: `test/test_interrogate_cache.py`
- Modify: `modules/interrogate.py`
- Modify: `CORE_IMPROVEMENT.md`

- [x] **Step 1: Write the failing test**

Add this test to `test/test_interrogate_cache.py`:

```python
def test_rank_batches_similarity_for_multiple_image_features(monkeypatch):
    model, _fake_clip = create_interrogator()
    image_features = torch.tensor([[1.0, 0.5], [0.25, 1.5]], dtype=torch.float32)
    text_items = ["first", "second", "third"]
    _limited_text_array, text_features = model.cached_text_features(text_items, "artists")
    expected_similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1).mean(dim=0, keepdim=True)
    expected_probs, expected_labels = expected_similarity.cpu().topk(2, dim=-1)
    expected = [
        (text_items[expected_labels[0][i].numpy()], expected_probs[0][i].numpy() * 100)
        for i in range(2)
    ]
    softmax_shapes = []
    original_softmax = torch.Tensor.softmax

    def recording_softmax(self, *args, **kwargs):
        softmax_shapes.append(tuple(self.shape))
        return original_softmax(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "softmax", recording_softmax)

    actual = model.rank(image_features, text_items, top_count=2, category_name="artists")

    assert actual == expected
    assert softmax_shapes == [(2, 3)]
```

- [x] **Step 2: Run test to verify it fails**

Run:

```powershell
D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_interrogate_cache.py::test_rank_batches_similarity_for_multiple_image_features -q
```

Expected: FAIL because current code records two softmax calls with shape `(1, 3)`.

- [x] **Step 3: Write minimal implementation**

Update `modules/interrogate.py`:

```python
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
similarity = similarity.mean(dim=0, keepdim=True)
```

- [x] **Step 4: Mark improvement done**

Update row 19 in `CORE_IMPROVEMENT.md` from `backlog` to `done`.

- [x] **Step 5: Run focused and regression verification**

Run:

```powershell
D:\stable-diffusion\envs\v1101\python.exe -m py_compile modules\interrogate.py test\test_interrogate_cache.py
D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_interrogate_cache.py -q
D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_ui_common_save_files.py test/test_outpainting_mk2.py test/test_sd_models_lookup.py test/test_cache.py test/test_ui_extra_networks_paging.py -q
D:\stable-diffusion\envs\v1101\python.exe -c "import sys, unittest; sys.argv=['unittest']; names=['test.test_sd_hijack_clip_tensor_cache','test.test_prompt_parser_reconstruction','test.test_cfg_denoiser_inputs','test.test_api_image_inputs','test.test_hashes','test.test_sd_models_lookup','test.test_image_hash','test.test_extensions_repo_metadata_cache']; suite=unittest.defaultTestLoader.loadTestsFromNames(names); result=unittest.TextTestRunner(verbosity=2).run(suite); raise SystemExit(0 if result.wasSuccessful() else 1)"
git diff --check
```

Expected: all tests pass; `git diff --check` has no whitespace errors, though Git may report existing CRLF warnings.
