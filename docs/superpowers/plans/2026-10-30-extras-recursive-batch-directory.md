# Recursive Extras Batch Directory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Extras → Batch from Directory process nested images recursively and recreate their relative directory structure under the active output root.

**Architecture:** Extend the shared recursive file walker with backward-compatible excluded-subtree pruning, then have Extras directory mode attach a relative parent path to each work item. Use an item-specific destination directory for skip checks, image saves, extra outputs, and captions while leaving other Extras modes unchanged.

**Tech Stack:** Python 3, Pillow, pytest, Gradio, existing `modules.util.walk_files()` and Extras postprocessing pipeline.

**Spec:** `docs/superpowers/specs/2026-10-30-extras-recursive-batch-directory-design.md`

## Global Constraints

- Recursion is always enabled for Extras → Batch from Directory; no new toggle is added.
- Explicit and default Extras output roots both preserve source-relative parent directories.
- An output root strictly beneath the input root is excluded from discovery; an output root equal to the input root is not excluded.
- Existing callers of `walk_files()` and existing Single Image and Batch Process behavior remain unchanged.
- Unreadable and non-image files continue to be skipped by the existing image-loading path.
- Empty source directories are not copied.

## Review Focus

- Two source files with the same basename in different subdirectories must save independently in their mirrored destinations; covered by Task 2's recursive-tree test.
- A default output root must mirror paths exactly like an explicit output root; covered by Task 2's default-root test.
- A pre-existing output subtree beneath the input root must not enter the manifest or job count; covered by Task 1's pruning tests and Task 2's nested-output test.
- Input and output resolving to the same directory must still process source images without excluding the input root; covered by Task 2's same-root test.
- Skip-existing checks must use the mirrored directory and effective postprocessor output extension; covered by Task 2's nested skip-existing test.

---

### Task 1: Add excluded-subtree support to the shared file walker

**Files:**
- Modify: `modules/util.py:31-74`
- Test: `test/test_walk_files.py`

**Interfaces:**
- Consumes: Existing `walk_files(path, allowed_extensions=None)` callers and `shared.opts.list_hidden_files` policy.
- Produces: `walk_files(path, allowed_extensions=None, excluded_paths=None)`, yielding the same naturally sorted file paths while pruning canonical excluded directory trees.

- [ ] **Step 1: Write failing tests for excluded subtree pruning**

Add tests that build a temporary tree and assert:

```python
assert list(util.walk_files(root, excluded_paths=[root / "output"])) == [str(root / "source.png")]
```

Also cover an excluded path referenced through a symlink when the platform permits symlink creation, and confirm the existing lazy-yield/extension-filter test remains valid when `excluded_paths` is omitted.

- [ ] **Step 2: Run the walker tests and verify the new tests fail**

Run: `& 'D:\stable-diffusion\envs\v1101\python.exe' -m pytest test\test_walk_files.py -q`

Expected: FAIL because `walk_files()` does not accept `excluded_paths` or still yields files beneath the excluded tree.

- [ ] **Step 3: Implement canonical excluded-tree pruning**

Change the signature in `modules/util.py` to:

```python
def walk_files(path, allowed_extensions=None, excluded_paths=None):
```

Normalize exclusions with `os.path.realpath()` and `os.path.normcase()`. Before yielding a root or descending into a child directory, prune paths equal to or beneath an excluded root using `os.path.commonpath()` with cross-drive `ValueError` handling. Preserve hidden-directory handling, natural sorting, symlink following, visited-directory cycle detection, laziness, and extension filtering.

- [ ] **Step 4: Run the walker tests and verify they pass**

Run: `& 'D:\stable-diffusion\envs\v1101\python.exe' -m pytest test\test_walk_files.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the walker change**

```bash
git add modules/util.py test/test_walk_files.py
git commit -m "feat: allow recursive file walker exclusions"
```

### Task 2: Mirror recursive Extras directory outputs

**Files:**
- Modify: `modules/postprocessing.py:16-157`
- Modify: `modules/ui_postprocessing.py:20-23`
- Test: `test/test_extras.py`

**Interfaces:**
- Consumes: `shared.walk_files(input_dir, excluded_paths=...)` from Task 1 and the existing `run_postprocessing()` arguments.
- Produces: Directory-mode work items carrying `(image_placeholder, source_name, relative_parent)` and per-item output directories used by preflight and save operations. `run_postprocessing()` retains its existing public signature and return tuple.

- [ ] **Step 1: Write failing recursive Extras tests**

Add focused tests to `test/test_extras.py` that assert:

```python
saved_paths == {
    output_root,
    output_root / "people",
    output_root / "animals",
}
```

Use duplicate `portrait.png` basenames under `people/` and `animals/`. Add separate tests for the configured default output root, a nested output subtree excluded from processing and job count, input equal to output, and an existing `nested/source.png` matching the effective postprocessor output extension being skipped before scripts run. Update the unit-test option fixture with `list_hidden_files=False` so the real walker can be exercised.

- [ ] **Step 2: Run focused Extras tests and verify the new tests fail**

Run: `& 'D:\stable-diffusion\envs\v1101\python.exe' -m pytest test\test_extras.py -q -k 'not simple_upscaling_performed and not png_info_performed and not interrogate_performed'`

Expected: FAIL because directory mode is flat and non-recursive.

- [ ] **Step 3: Add relative-parent work items and strict descendant detection**

In `modules/postprocessing.py`, add a private helper with this interface:

```python
def _is_strict_subpath(path: str, parent: str) -> bool:
```

Resolve both paths through `realpath()` and `normcase()`, return false when equal, and use `commonpath()` with cross-drive handling. Compute the active `outpath` before building the manifest. In directory mode, pass `[outpath]` as `excluded_paths` only when `_is_strict_subpath(outpath, input_dir)` is true. Yield each file with its lexical parent relative to `input_dir`; use an empty relative parent for root-level files and all non-directory modes.

- [ ] **Step 4: Use the item-specific output directory everywhere**

Change the private existence helper to:

```python
def output_exists(item_outpath, basename, suffix="", extension=None):
```

For each work item, derive `item_outpath = os.path.join(outpath, relative_parent)` when the relative parent is non-empty. Use it for initial skip checks, per-extra-image skip checks, and `images.save_image(path=item_outpath, ...)`. Caption paths continue to derive from the returned saved filename, keeping them beside their corresponding image.

- [ ] **Step 5: Clarify recursive behavior in the UI**

Update the Batch from Directory input and output placeholders in `modules/ui_postprocessing.py` to state that subdirectories are processed recursively and recreated beneath the output directory.

- [ ] **Step 6: Run focused Extras tests and verify they pass**

Run: `& 'D:\stable-diffusion\envs\v1101\python.exe' -m pytest test\test_extras.py -q -k 'not simple_upscaling_performed and not png_info_performed and not interrogate_performed'`

Expected: PASS.

- [ ] **Step 7: Run the complete focused regression set**

Run: `& 'D:\stable-diffusion\envs\v1101\python.exe' -m pytest test\test_walk_files.py test\test_background_removal_config.py test\test_background_removal.py test\test_background_removal_script.py test\test_scripts_postprocessing.py test\test_extras.py -q -k 'not simple_upscaling_performed and not png_info_performed and not interrogate_performed'`

Expected: PASS with only existing dependency/configuration warnings.

- [ ] **Step 8: Check formatting and scope**

Run: `git diff --check`

Expected: no output. Review `git status --short` and confirm only the design, plan, utility, postprocessing, UI, and focused test files changed.

- [ ] **Step 9: Commit the Extras change**

```bash
git add modules/postprocessing.py modules/ui_postprocessing.py test/test_extras.py docs/superpowers/specs/2026-10-30-extras-recursive-batch-directory-design.md docs/superpowers/plans/2026-10-30-extras-recursive-batch-directory.md
git commit -m "feat: process Extras directories recursively"
```
