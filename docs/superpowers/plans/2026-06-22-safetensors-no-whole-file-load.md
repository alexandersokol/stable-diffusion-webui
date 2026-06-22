# Safetensors No Whole-File Load Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the whole-file Python bytes allocation from disabled-mmap `.safetensors` checkpoint loading.

**Architecture:** Keep `modules.sd_models.read_state_dict()` as the only production change. The `.safetensors` branch should resolve the target device once and call `safetensors.torch.load_file(checkpoint_file, device=device)` regardless of the `disable_mmap_load_safetensors` option, avoiding both `open(...).read()` and the extra tensor `.to(device)` pass.

**Tech Stack:** Python, safetensors `0.4.2`, existing unittest/pytest-compatible test harness.

---

### Task 1: Safetensors Loader Regression

**Files:**
- Modify: `test/test_sd_models_lookup.py`
- Modify: `modules/sd_models.py`
- Modify: `CORE_IMPROVEMENT.md`

- [x] **Step 1: Write the failing test**

Add a test to `test/test_sd_models_lookup.py` using `load_sd_models_for_test()`:

```python
def test_read_state_dict_uses_load_file_when_safetensors_mmap_is_disabled(self):
    sd_models = load_sd_models_for_test()
    sd_models.shared.opts.disable_mmap_load_safetensors = True
    load_file_calls = []

    class FakeTensor:
        pass

    def fake_load_file(filename, device):
        load_file_calls.append((filename, device))
        return {"weight": FakeTensor()}

    def fail_load(data):
        raise AssertionError("safetensors.torch.load should not receive whole-file bytes")

    def fail_open(*args, **kwargs):
        raise AssertionError("disabled mmap safetensors loading should not read the whole file into bytes")

    sd_models.safetensors.torch.load_file = fake_load_file
    sd_models.safetensors.torch.load = fail_load
    sd_models.open = fail_open

    state_dict = sd_models.read_state_dict("model.safetensors", map_location="cuda")

    self.assertEqual(load_file_calls, [("model.safetensors", "cuda")])
    self.assertEqual(list(state_dict), ["weight"])
```

- [x] **Step 2: Run test to verify it fails**

Run:

```powershell
D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_sd_models_lookup.py::test_read_state_dict_uses_load_file_when_safetensors_mmap_is_disabled -q
```

Expected: FAIL because the current disabled-mmap branch calls `open(checkpoint_file, 'rb').read()`.

- [x] **Step 3: Write minimal implementation**

Replace the `.safetensors` loader branch in `modules/sd_models.py` with a single `load_file()` call:

```python
if extension.lower() == ".safetensors":
    device = map_location or shared.weight_load_location or devices.get_optimal_device_name()
    pl_sd = safetensors.torch.load_file(checkpoint_file, device=device)
else:
    pl_sd = torch.load(checkpoint_file, map_location=map_location or shared.weight_load_location)
```

- [x] **Step 4: Mark improvement done**

Update row 16 in `CORE_IMPROVEMENT.md` from `backlog` to `done`.

- [x] **Step 5: Run focused and regression verification**

Run:

```powershell
D:\stable-diffusion\envs\v1101\python.exe -m py_compile modules\sd_models.py test\test_sd_models_lookup.py
D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_sd_models_lookup.py::test_read_state_dict_uses_load_file_when_safetensors_mmap_is_disabled test/test_sd_models_lookup.py -q
D:\stable-diffusion\envs\v1101\python.exe -m pytest test/test_cache.py test/test_ui_extra_networks_paging.py -q
D:\stable-diffusion\envs\v1101\python.exe -m unittest test.test_sd_hijack_clip_tensor_cache test.test_prompt_parser_reconstruction test.test_cfg_denoiser_inputs test.test_api_image_inputs test.test_hashes test.test_sd_models_lookup test.test_image_hash test.test_extensions_repo_metadata_cache -v
git diff --check
```

Expected: all tests pass; `git diff --check` has no whitespace errors, though Git may report existing CRLF warnings.
