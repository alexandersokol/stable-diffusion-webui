# Interrogate Text Feature Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Avoid repeated CLIP tokenization and text encoding for unchanged interrogate category text.

**Architecture:** Add a small cache owned by `InterrogateModels`, keyed by category identity, text limit, CLIP model identity, interrogate device, and dtype. `rank()` retrieves normalized text features from the cache and invalidates them when categories or the CLIP model context changes.

**Tech Stack:** Python, PyTorch tensors, existing WebUI interrogate module, focused direct Python verification.

---

### Task 1: Add Interrogate Text Feature Cache

**Files:**
- Modify: `modules/interrogate.py`

- [x] **Step 1: Track cache state on the interrogator**

Add `_text_feature_cache` and `_text_feature_cache_model_key` to `InterrogateModels.__init__`.

- [x] **Step 2: Clear cache when category files reload**

When `categories()` rebuilds `self.loaded_categories`, clear cached text features because category names or item arrays may have changed.

- [x] **Step 3: Clear cache when CLIP model context changes**

After `load()` establishes `self.clip_model`, `devices.device_interrogate`, and `self.dtype`, compare the current model key to the previous one and clear cached features if it changed.

- [x] **Step 4: Use cached features in `rank()`**

Replace repeated tokenization/encoding in `rank()` with a helper that returns normalized text features and the limited text array.

### Task 2: Add Focused Verification

**Files:**
- Create: `test/test_interrogate_cache.py`
- Modify: `IMPROVEMENT_PLAN.md`

- [x] **Step 1: Add direct tests**

Create tests that stub `clip.tokenize`, `clip_model.encode_text`, and `devices.device_interrogate`, then verify cache reuse and invalidation by dict limit/model key.

- [x] **Step 2: Mark issue #16 done**

Update the status in `IMPROVEMENT_PLAN.md` from `backlog` to `done`.

- [x] **Step 3: Run verification**

Run compile checks and direct Python test calls because `pytest` is not installed in the provided Conda environment.
