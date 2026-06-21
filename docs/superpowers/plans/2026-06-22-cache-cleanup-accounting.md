# Cache Cleanup Accounting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make disk cache cleanup avoid a full all-subsection usage scan after every single eviction.

**Architecture:** Keep the existing `cleanup_cache()` public return shape. Use one full cache usage scan before eviction and one after eviction for final reporting, while maintaining an in-memory usage map during the eviction loop from per-subsection volume deltas.

**Tech Stack:** Python, diskcache, pytest.

---

### Task 1: Reduce Cleanup Scan Count

**Files:**
- Modify: `test/test_cache.py`
- Modify: `modules/cache.py`
- Modify: `CORE_IMPROVEMENT.md`

- [x] **Step 1: Write the failing test**

Add a pytest test to `test/test_cache.py` that creates several oversized cache entries, patches `cache.get_cache_usage()` to count full usage scans, runs `cleanup_cache()` with a low budget, and asserts the cleanup evicts multiple entries while calling `get_cache_usage()` only twice.

- [x] **Step 2: Run test to verify it fails**

Run: `D:\stable-diffusion\envs\v1101\python.exe -c "import test.test_cache as t; t.test_cleanup_cache_does_not_rescan_all_subsections_after_each_eviction()"`

Expected: FAIL because current `cleanup_cache()` calls `get_cache_usage()` once before eviction, once after every eviction, and once after the loop.

- [x] **Step 3: Implement minimal accounting change**

Change `_pop_oldest_cache_entry()` to return `(evicted, removed_bytes)` using the subsection cache volume before and after the pop. Change `cleanup_cache()` to update `usage[subsection]` and `total` from that delta instead of rescanning every subsection inside the loop.

- [x] **Step 4: Verify focused tests pass**

Run: `D:\stable-diffusion\envs\v1101\python.exe -c "import test.test_cache as t; [getattr(t, name)() for name in dir(t) if name.startswith('test_')]"`

Expected: PASS.

- [x] **Step 5: Run regression checks**

Run the cache tests, compile `modules/cache.py`, run nearby prior core tests, and run `git diff --check`.
