# Tiled Upscale CPU Accumulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce GPU/accelerator memory pressure during tiled upscaling by moving full-size accumulation buffers to CPU.

**Architecture:** Keep per-tile model inference on the model device, then copy each tile output to CPU for weighted accumulation. Return the CPU output tensor to existing PIL conversion callers, which already move tensors to CPU.

**Tech Stack:** Python, PyTorch tensors, existing WebUI upscaler utilities, focused direct Python verification.

---

### Task 1: Move Tiled Accumulation To CPU

**Files:**
- Modify: `modules/upscaler_utils.py`

- [x] **Step 1: Allocate full-size buffers on CPU**

Change `tiled_upscale_2()` so `result` and `weights` are created on `torch.device("cpu")` instead of the model device.

- [x] **Step 2: Accumulate model output on CPU**

After `out_patch = model(in_patch)`, detach and move `out_patch` to the accumulation device before adding it to `result`.

- [x] **Step 3: Avoid per-tile mask allocation**

Replace `out_patch_mask = torch.ones_like(out_patch)` with adding scalar `1` to the corresponding `weights` slice.

### Task 2: Verify Behavior

**Files:**
- Create: `test/test_upscaler_utils.py`
- Modify: `IMPROVEMENT_PLAN.md`
- Modify: `docs/superpowers/plans/2026-06-21-tiled-upscale-cpu-accumulation.md`

- [x] **Step 1: Add focused tests**

Add direct tests that call `tiled_upscale_2()` with a fake model and verify the tiled result matches expected weighted accumulation and returns a CPU tensor.

- [x] **Step 2: Mark issue #17 done**

Update `IMPROVEMENT_PLAN.md` status for issue #17 from `backlog` to `done`.

- [x] **Step 3: Run verification**

Run `compileall`, direct Python tests, and `git diff --check`.
