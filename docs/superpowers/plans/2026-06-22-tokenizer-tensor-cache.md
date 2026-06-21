# Tokenizer Tensor Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reuse prompt token and multiplier tensors across repeated CLIP conditioning calls when the prompt chunk and conditioning options are unchanged.

**Architecture:** Add an instance-local bounded LRU cache to `TextConditionalModel`. The cache stores only tensors derived from token ids and multipliers; transformer outputs are still recomputed every call so textual inversion fixes, pooled output, and emphasis logic remain fresh.

**Tech Stack:** Python, PyTorch, unittest.

---

### Task 1: Add Tensor Cache Coverage

**Files:**
- Create: `test/test_sd_hijack_clip_tensor_cache.py`
- Modify: `modules/sd_hijack_clip.py`
- Modify: `CORE_IMPROVEMENT.md`

- [x] **Step 1: Write the failing test**

Create tests using a minimal `TextConditionalModel` subclass that counts encode calls and patches `torch.asarray`. The first test calls `process_tokens()` twice with the same token and multiplier lists and expects only two tensor materializations total: one for tokens and one for multipliers.

- [x] **Step 2: Run test to verify it fails**

Run: `D:\stable-diffusion\envs\v1101\python.exe -m unittest test.test_sd_hijack_clip_tensor_cache -v`

Expected: FAIL because repeated calls currently call `torch.asarray()` four times.

- [x] **Step 3: Implement minimal cache**

In `TextConditionalModel`, add an `OrderedDict` LRU cache with a small fixed max size. Build the cache key from token ids, multipliers, active device, ids for end/pad tokens, emphasis, CLIP skip options, tokenizer identity, and embedding-db identity/signature. Use the cached tensors in `process_tokens()` and still call `encode_with_transformers()` every time.

- [x] **Step 4: Verify focused tests pass**

Run: `D:\stable-diffusion\envs\v1101\python.exe -m unittest test.test_sd_hijack_clip_tensor_cache -v`

Expected: PASS.

- [x] **Step 5: Run regression checks**

Run the nearby prompt/conditioning tests, compile changed files, and run `git diff --check`.
