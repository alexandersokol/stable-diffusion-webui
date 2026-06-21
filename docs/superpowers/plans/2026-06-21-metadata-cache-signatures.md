# Metadata Cache Signatures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make file-backed metadata cache entries more reliable and prevent duplicate expensive metadata reads during concurrent refreshes.

**Architecture:** Strengthen `modules.cache.cached_data_for_file()` by storing a file signature with `mtime`, `mtime_ns`, and `size`, while still reading older `mtime`-only entries. Add a per-cache-entry lock around miss generation so simultaneous callers compute each file’s metadata once.

**Tech Stack:** Python, `diskcache`, `threading`, existing WebUI cache helpers, direct Python verification.

---

### Task 1: Harden Cached File Entries

**Files:**
- Modify: `modules/cache.py`

- [x] **Step 1: Add file signature helper**

Create a helper that returns a dict containing file `mtime`, `mtime_ns`, and `size` from `os.stat()`.

- [x] **Step 2: Add cache entry freshness helper**

Compare new signatures to entries, accepting old entries that only have `mtime` when the file mtime still matches.

- [x] **Step 3: Add per-file miss locks**

Create a lock table keyed by `(subsection, title, filename)` so only one thread runs `func()` for the same cache miss.

- [x] **Step 4: Update `cached_data_for_file()`**

Use signature freshness and miss locks while preserving existing return behavior and cleanup calls after writes.

### Task 2: Add Verification

**Files:**
- Modify: `test/test_cache.py`
- Modify: `IMPROVEMENT_PLAN.md`
- Modify: `docs/superpowers/plans/2026-06-21-metadata-cache-signatures.md`

- [x] **Step 1: Add direct tests**

Add tests for size-based invalidation, legacy mtime-only cache compatibility, and concurrent miss deduplication.

- [x] **Step 2: Mark issue #19 done**

Update `IMPROVEMENT_PLAN.md` issue #19 from `backlog` to `done`.

- [x] **Step 3: Run verification**

Run `compileall`, direct Python cache tests, and `git diff --check`.
