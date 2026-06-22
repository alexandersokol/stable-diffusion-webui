# Checkpoint UI Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the main `Stable Diffusion checkpoint` UI selector display only checkpoint file names without extensions or hashes while preserving internal checkpoint values.

**Architecture:** Keep `CheckpointInfo.title`, aliases, API responses, merger dropdowns, and model loading untouched. Add a UI-only helper that returns Gradio `(label, value)` pairs where the label is `CheckpointInfo.name_for_extra` and the value is the current internal title.

**Tech Stack:** Python, Gradio dropdown choices, pytest/unittest.

---

### Task 1: UI Choice Formatter

**Files:**
- Modify: `modules/sd_models.py`
- Modify: `modules/shared_items.py`
- Modify: `modules/shared_options.py`
- Test: `test/test_sd_models_lookup.py`

- [ ] **Step 1: Write failing test**

Add a test that constructs a fake checkpoint with:

```python
info.name_for_extra = "my_model"
info.title = "models/subdir/my_model.safetensors [abc123def0]"
```

Then asserts `sd_models.checkpoint_tiles_for_ui()` returns:

```python
[("my_model", "models/subdir/my_model.safetensors [abc123def0]")]
```

- [ ] **Step 2: Verify RED**

Run: `D:\stable-diffusion\envs\v1101\python.exe -m pytest test\test_sd_models_lookup.py::test_checkpoint_tiles_for_ui_hide_extension_and_hash -q`

Expected: fail because `checkpoint_tiles_for_ui` does not exist.

- [ ] **Step 3: Implement helper and wire main setting**

Add `checkpoint_tiles_for_ui()` to `modules/sd_models.py`, expose it as `shared_items.list_checkpoint_tiles_for_ui()`, and update only `shared_options.py` `sd_model_checkpoint` choices lambda to use it.

- [ ] **Step 4: Verify GREEN**

Run the focused test, then the existing checkpoint tests.
