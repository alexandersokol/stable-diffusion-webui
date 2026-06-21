# Adaptive Torch GC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce generation stalls from repeated backend cache cleanup while still freeing memory when pressure is high.

**Architecture:** Keep `devices.torch_gc()` as the central API and add adaptive skip logic inside it. CUDA cleanup runs when forced, when enough time has elapsed, or when memory/cache pressure crosses conservative thresholds; other accelerator backends use a time throttle unless forced.

**Tech Stack:** Python, PyTorch backend APIs, existing WebUI options, direct Python verification.

---

### Task 1: Add Adaptive GC Policy

**Files:**
- Modify: `modules/devices.py`
- Modify: `modules/shared_options.py`

- [x] **Step 1: Add GC policy helpers**

Add helpers to read `torch_gc_min_interval`, `torch_gc_cuda_free_memory_threshold`, and `torch_gc_cuda_reserved_memory_threshold` from `shared.opts` with safe defaults.

- [x] **Step 2: Add CUDA pressure checks**

Use `torch.cuda.mem_get_info()`, `torch.cuda.memory_reserved()`, and `torch.cuda.memory_allocated()` to decide whether cleanup should run despite the interval throttle.

- [x] **Step 3: Preserve force cleanup**

Allow `devices.torch_gc(force=True)` to bypass all throttles and cleanup each available backend.

- [x] **Step 4: Add settings**

Add conservative settings in the Optimization section so users can tune or disable adaptive throttling.

### Task 2: Add Verification

**Files:**
- Create: `test/test_devices_gc.py`
- Modify: `IMPROVEMENT_PLAN.md`
- Modify: `docs/superpowers/plans/2026-06-21-adaptive-torch-gc.md`

- [x] **Step 1: Add direct tests**

Add tests for recent healthy CUDA skip, forced cleanup, CUDA memory pressure cleanup, and non-CUDA backend throttling.

- [x] **Step 2: Mark issue #18 done**

Update `IMPROVEMENT_PLAN.md` issue #18 from `backlog` to `done`.

- [x] **Step 3: Run verification**

Run `compileall`, direct Python tests, and `git diff --check`.
