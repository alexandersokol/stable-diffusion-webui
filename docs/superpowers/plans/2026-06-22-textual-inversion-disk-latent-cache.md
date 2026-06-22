# Textual Inversion Disk Latent Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in disk-backed latent tensor cache for textual inversion datasets to reduce steady-state training dataset RAM usage.

**Architecture:** Keep the current memory-backed path as the default. Add a `memory`/`disk` training option and optional constructor parameters for tests; in disk mode, dataset construction computes tensors exactly as before, writes them to a temporary cache file, clears tensor fields from the stored dataset entry, and `__getitem__` returns a short-lived hydrated copy for batching.

**Tech Stack:** Python, PyTorch tensor serialization, pytest, existing Stable Diffusion WebUI training settings.

---

### Task 1: Failing Tests For Disk Cache Behavior

**Files:**
- Create: `test/test_textual_inversion_dataset_disk_cache.py`
- Modify: none

- [ ] **Step 1: Write failing tests**

Add tests that create a tiny temporary image dataset, instantiate `modules.textual_inversion.dataset.PersonalizedBase`, and assert:

```python
def test_memory_mode_keeps_latent_sample_on_dataset_entry(tmp_path):
    ds = build_dataset(tmp_path, cache_mode="memory")
    assert ds.dataset[0].latent_sample is not None
    assert ds[0].latent_sample is ds.dataset[0].latent_sample


def test_disk_mode_spills_latent_sample_and_hydrates_copy(tmp_path):
    ds = build_dataset(tmp_path, cache_mode="disk", cache_dir=tmp_path / "cache")
    stored_entry = ds.dataset[0]
    assert stored_entry.latent_sample is None
    assert stored_entry.cache_file is not None

    loaded_entry = ds[0]
    assert loaded_entry is not stored_entry
    assert torch.equal(loaded_entry.latent_sample, expected_latent())
    assert stored_entry.latent_sample is None


def test_disk_mode_spills_conditioning_and_weights(tmp_path):
    ds = build_dataset(tmp_path, cache_mode="disk", cache_dir=tmp_path / "cache", include_cond=True, use_weight=True, rgba=True)
    stored_entry = ds.dataset[0]
    assert stored_entry.cond is None
    assert stored_entry.weight is None

    loaded_entry = ds[0]
    assert torch.equal(loaded_entry.cond, torch.tensor([4.0, 5.0]))
    assert loaded_entry.weight is not None


def test_disk_mode_random_sampling_loads_distribution_without_retaining_it(tmp_path, monkeypatch):
    fake_model = FakeModel()
    monkeypatch.setattr(dataset_module.shared, "sd_model", fake_model)
    ds = build_dataset(tmp_path, cache_mode="disk", cache_dir=tmp_path / "cache", latent_sampling_method="random", model=fake_model)
    stored_entry = ds.dataset[0]
    assert stored_entry.latent_dist is None

    first = ds[0]
    second = ds[0]
    assert torch.equal(first.latent_sample, expected_latent(2.0))
    assert torch.equal(second.latent_sample, expected_latent(3.0))
    assert stored_entry.latent_dist is None
```

- [ ] **Step 2: Run tests to verify RED**

Run: `D:\stable-diffusion\envs\v1101\python.exe -m pytest test\test_textual_inversion_dataset_disk_cache.py -q`

Expected: FAIL because `PersonalizedBase.__init__` does not accept `cache_mode`/`cache_dir` and `DatasetEntry` has no `cache_file`.

### Task 2: Disk Cache Implementation

**Files:**
- Modify: `modules/textual_inversion/dataset.py`
- Modify: `modules/shared_options.py`
- Test: `test/test_textual_inversion_dataset_disk_cache.py`

- [ ] **Step 1: Add cache metadata and helpers**

In `modules/textual_inversion/dataset.py`, import `copy`, `tempfile`, and `shutil`. Add `cache_file=None` to `DatasetEntry.__init__`. Add `DatasetEntry.copy_metadata()` returning a new `DatasetEntry` with filename, filename_text, cond_text, pixel_values, and cache_file copied.

- [ ] **Step 2: Add constructor controls**

Extend `PersonalizedBase.__init__` with `cache_mode=None, cache_dir=None`. Resolve `cache_mode` from `shared.opts.training_dataset_latent_cache_mode` when omitted, defaulting to `"memory"`. In disk mode, create `cache_dir` or a `tempfile.mkdtemp(prefix="textual-inversion-latents-")` directory and remember whether cleanup is owned.

- [ ] **Step 3: Spill tensors during dataset preparation**

After each `DatasetEntry` is populated, write a dictionary containing any non-None `latent_sample`, `latent_dist`, `cond`, and `weight` to `entry.cache_file` with `torch.save`. Clear those tensor attributes on the stored entry.

- [ ] **Step 4: Hydrate entries on access**

In `__getitem__`, for disk mode, create a metadata copy, load cached tensors with `torch.load(map_location=devices.cpu)`, apply dynamic tag text when needed, and compute fresh random latent samples from the loaded distribution. Return the copy so stored entries remain tensor-free.

- [ ] **Step 5: Add training option**

In `modules/shared_options.py`, add:

```python
"training_dataset_latent_cache_mode": OptionInfo("memory", "Textual inversion dataset latent cache", gr.Radio, {"choices": ["memory", "disk"]}).info("memory keeps current behavior; disk lowers RAM use by loading cached tensors per batch"),
```

- [ ] **Step 6: Run focused tests to verify GREEN**

Run: `D:\stable-diffusion\envs\v1101\python.exe -m pytest test\test_textual_inversion_dataset_disk_cache.py -q`

Expected: PASS.

### Task 3: Regression Verification And Status

**Files:**
- Modify: `CORE_IMPROVEMENT.md`

- [ ] **Step 1: Mark task 22 done**

Change row `22` status from `backlog` to `done`.

- [ ] **Step 2: Run syntax and regression checks**

Run:

```powershell
D:\stable-diffusion\envs\v1101\python.exe -m py_compile modules\textual_inversion\dataset.py modules\shared_options.py test\test_textual_inversion_dataset_disk_cache.py
D:\stable-diffusion\envs\v1101\python.exe -m pytest test\test_textual_inversion_dataset_disk_cache.py -q
D:\stable-diffusion\envs\v1101\python.exe -m pytest test\test_textual_inversion_image_embedding.py test\test_progress.py test\test_interrogate_cache.py test\test_ui_common_save_files.py test\test_outpainting_mk2.py test\test_sd_models_lookup.py -q
```

Expected: all commands pass.
