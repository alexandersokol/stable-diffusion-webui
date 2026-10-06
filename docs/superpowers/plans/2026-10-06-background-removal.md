# Extras Background Removal Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Use `superpowers:test-driven-development` for every behavior change and `superpowers:verification-before-completion` before claiming completion.

**Goal:** Add an optional, portable Background Removal operation to Extras that supports all 13 supplied ONNX models, runs last, exposes only installed models, works through UI and API inputs, and forces transparent outputs to PNG.

**Architecture:** Add a registry-driven `modules/background_removal.py` subsystem with architecture-specific adapters and a one-session runtime cache. Expose it through a terminal Extras script. Extend the postprocessing framework with terminal ordering and an output-format preflight so skip-existing and saving agree before inference. Keep ONNX Runtime lazily imported and make CPU ONNX Runtime the default installed distribution.

**Tech stack:** Python 3.10, Pillow, NumPy, ONNX Runtime 1.21.x, Gradio 3.41, Pydantic/FastAPI, pytest.

**Approved design:** `docs/superpowers/specs/2026-10-06-background-removal-design.md`

---

## Task 1: Add the model-path contract and portable dependency

**Files:**

- Modify: `modules/cmd_args.py`
- Modify: `requirements.txt`
- Modify: `requirements_versions.txt`
- Create: `test/test_background_removal_config.py`

### Step 1: Write failing configuration tests

Add tests that import the parsed command options and assert:

```python
assert shared.cmd_opts.background_removal_models_path == os.path.join(models_path, "bg-removal")
```

Add a parser-level test with a temporary absolute path and assert the new argument is normalized consistently with the existing `--esrgan-models-path` option.

Run:

```powershell
.\venv\Scripts\python.exe -m pytest test\test_background_removal_config.py -q
```

Expected: failure because the option does not exist.

### Step 2: Add the CLI option

Near the existing upscaler model paths in `modules/cmd_args.py`, add:

```python
parser.add_argument(
    "--background-removal-models-path",
    type=normalized_filepath,
    help="Path to directory with background-removal ONNX model files.",
    default=os.path.join(models_path, "bg-removal"),
)
```

### Step 3: Add the baseline ONNX Runtime dependency

Add `onnxruntime` to `requirements.txt` and pin the CPU baseline to `onnxruntime==1.21.0` in `requirements_versions.txt`, matching the version already exercised by the source project. Do not add `onnxruntime-gpu` or `onnxruntime-directml`; documentation will explain that users replace the CPU distribution with exactly one provider-specific distribution when desired.

### Step 4: Run the focused tests

Run the command from Step 1. Expected: pass.

### Step 5: Commit

```powershell
git add modules/cmd_args.py requirements.txt requirements_versions.txt test/test_background_removal_config.py
git commit -m "feat: configure background removal model path"
```

## Task 2: Build the exact 13-model registry and discovery

**Files:**

- Create: `modules/background_removal.py`
- Create: `test/test_background_removal.py`

### Step 1: Write failing registry tests

Test that the immutable registry has exactly these filenames and unique display names:

```text
anime-seg.onnx
BEN2_Base.onnx
BEN2_ONNX.onnx
BiRefNet-DIS.onnx
BiRefNet-general.onnx
BiRefNet-massive.onnx
BiRefNet-portrait.onnx
deeplabv3-mobilevit-small.onnx
isnet-anime.onnx
RMBG-1.4.onnx
RMBG-2.0.onnx
silueta.onnx
u2net_human_seg.onnx
```

Use `tmp_path` to test empty discovery, partial installation, recursive discovery, unknown `.onnx` exclusion, separate anime entries, and deterministic duplicate handling with a captured warning.

Run:

```powershell
.\venv\Scripts\python.exe -m pytest test\test_background_removal.py -q
```

Expected: import failure because the module does not exist.

### Step 2: Define model contracts

In `modules/background_removal.py`, add frozen data classes/enums for:

```python
@dataclasses.dataclass(frozen=True)
class ModelSpec:
    display_name: str
    filename: str
    adapter: str
    input_size: tuple[int, int]
    input_name: str
    output_name: str
    resize_mode: str
    normalization: str
    activation: str
```

Create `MODEL_SPECS` with all 13 files. Encode known differences explicitly: 1024-pixel anime/BEN2/BiRefNet/RMBG/DeepLab inputs, 320-pixel U2Net/Silueta inputs, ImageNet versus `[0, 1]` normalization, letterbox versus stretch, and logits versus alpha output.

### Step 3: Implement installed-only discovery

Implement:

```python
def discover_models(root: str | os.PathLike[str]) -> dict[str, InstalledModel]:
    ...
```

Requirements:

- Resolve the configured root without creating it.
- Recursively scan for exact registered basenames.
- Ignore unknown files and unavailable entries.
- Sort candidate paths case-insensitively for deterministic selection.
- Warn when the same registered basename occurs more than once.
- Return entries keyed by stable display name for UI/API reuse.

### Step 4: Run focused tests and commit

Run the Task 2 test command. Expected: pass.

```powershell
git add modules/background_removal.py test/test_background_removal.py
git commit -m "feat: register background removal models"
```

## Task 3: Implement adapter preprocessing and mask reconstruction

**Files:**

- Modify: `modules/background_removal.py`
- Modify: `test/test_background_removal.py`

### Step 1: Write failing adapter-family tests

Use small synthetic Pillow images and reduced test-only `ModelSpec` sizes. Cover:

- RGB conversion and NCHW `float32` output.
- `[0, 1]` scaling.
- ImageNet mean/std normalization.
- Aspect-preserving square letterbox metadata and exact unpadding.
- Fixed stretching and restoration to the original dimensions.
- Output shapes `NCHW`, `NHW`, and `HW` where allowed.
- Sigmoid for logits and min/max normalization only where specified.
- Rejection of multi-class/incompatible output, NaN, infinity, and constant masks when normalization would be undefined.
- RGBA composition that leaves original RGB bytes unchanged.

Expected first run: failures because adapter functions do not exist.

### Step 2: Add pure preprocessing helpers

Implement Pillow/NumPy-only helpers that return both the tensor and resize metadata. Keep model-session concerns out of these functions so every architecture contract can be unit tested without ONNX Runtime.

### Step 3: Add output extraction and validation

Implement an adapter dispatch layer that:

- Selects the named graph output, with a clear contract error when absent.
- Reduces only supported singleton batch/channel dimensions.
- Applies the model-specific activation/normalization.
- Validates finite, meaningful alpha data.
- Reverses letterboxing or resizing and emits an 8-bit `L` mask at exact source dimensions.

### Step 4: Add RGBA composition

Convert the postprocessed input to RGB only for model input; preserve the image's RGB pixel values and attach the generated `L` mask as alpha.

### Step 5: Run and commit

```powershell
.\venv\Scripts\python.exe -m pytest test\test_background_removal.py -q
git add modules/background_removal.py test/test_background_removal.py
git commit -m "feat: add background removal adapters"
```

## Task 4: Add provider selection and the one-session runtime cache

**Files:**

- Modify: `modules/background_removal.py`
- Modify: `test/test_background_removal.py`

### Step 1: Write failing runtime tests

Inject a fake ONNX Runtime module/session factory and verify:

- ONNX Runtime is imported only when inference is requested.
- Available accelerated providers are preferred while CPU remains last.
- Unsupported providers are never requested.
- Session creation retries exactly once with CPU after provider initialization failure.
- The same `(path, size, mtime, providers)` identity reuses a session through a batch.
- Switching model or modifying the file invalidates and releases the cached session.
- Concurrent calls cannot create two sessions for one cache identity.
- Graph input/output name and tensor-shape mismatches include model name and path in the exception.

### Step 2: Implement the runtime service

Add `BackgroundRemovalService` with an injectable runtime loader/session factory for tests. Public methods:

```python
def remove_background(self, image: Image.Image, model: InstalledModel) -> Image.Image:
    ...

def unload(self) -> None:
    ...
```

Use one `threading.RLock` around cache mutation and inference. Log resolved model and actual session providers once after creation. Delete the old session reference before replacing it.

### Step 3: Run and commit

```powershell
.\venv\Scripts\python.exe -m pytest test\test_background_removal.py -q
git add modules/background_removal.py test/test_background_removal.py
git commit -m "feat: run background removal with ONNX"
```

## Task 5: Extend the postprocessing framework for terminal scripts and format preflight

**Files:**

- Modify: `modules/scripts_postprocessing.py`
- Create: `test/test_scripts_postprocessing.py`

### Step 1: Write failing framework tests

Create tiny fake postprocessing scripts and verify:

- Normal scripts retain preference/order behavior.
- `terminal = True` scripts always sort after non-terminal scripts even when the user preference lists them first.
- Multiple terminal scripts retain deterministic preference/order/name/index ordering.
- `output_extension(default_extension, **args)` is called in final script order.
- The default hook leaves the extension unchanged.
- `PostprocessedImage.create_copy()` preserves its preferred output extension.

### Step 2: Add framework contracts

Add to `ScriptPostprocessing`:

```python
terminal = False

def output_extension(self, default_extension, **args):
    return default_extension
```

Add `output_extension = None` to `PostprocessedImage` and copy it in `create_copy()`.

Refactor argument preparation into one private helper used by both `run()` and a new runner-level `output_extension(args, default_extension)` method. Add `terminal` as the first sort-key component so non-terminal scripts always precede terminal scripts.

### Step 3: Run and commit

```powershell
.\venv\Scripts\python.exe -m pytest test\test_scripts_postprocessing.py -q
git add modules/scripts_postprocessing.py test/test_scripts_postprocessing.py
git commit -m "feat: support terminal Extras processors"
```

## Task 6: Add the Background Removal Extras script and UI

**Files:**

- Create: `scripts/postprocessing_background_removal.py`
- Create: `test/test_background_removal_script.py`

### Step 1: Write failing script tests

Test the script class without starting the whole WebUI:

- Name is `Background Removal`, order is above existing processors, and `terminal` is true.
- Installed choices come from `shared.cmd_opts.background_removal_models_path`.
- Disabled processing is a no-op.
- Enabled processing rejects an empty, unknown, or disappeared model.
- Enabled processing calls the service, replaces `pp.image` with RGBA, adds useful postprocessing info, and sets `pp.output_extension = "png"`.
- Its preflight hook returns `png` only when enabled.

### Step 2: Implement the script behavior

Use `InputAccordion(False, label="Background Removal")` when at least one model is installed. Put the exact installed display names in a dropdown and show the resolved model directory plus a PNG transparency note.

When no model exists, render a closed explanatory accordion with a hidden disabled boolean input and an empty disabled dropdown. Keep argument keys identical in both UI branches so API argument creation remains stable.

### Step 3: Run and commit

```powershell
.\venv\Scripts\python.exe -m pytest test\test_background_removal_script.py -q
git add scripts/postprocessing_background_removal.py test/test_background_removal_script.py
git commit -m "feat: add Extras background removal UI"
```

## Task 7: Honor effective PNG format in skip-existing and saving

**Files:**

- Modify: `modules/postprocessing.py`
- Modify: `test/test_extras.py`

### Step 1: Write failing pipeline tests

Extend the existing unit-test fixture so the fake runner implements both `run()` and `output_extension()`. Verify:

- An enabled background-removal preflight changes a requested JPG/WebP format to PNG.
- Initial skip-existing checks look for `source.png`, not `source.jpg`.
- A pre-existing JPG does not skip a run that will emit PNG.
- Save calls receive `extension="png"` and no JPEG quality.
- Per-image preferred output extensions apply to extra images and their suffix checks.
- Disabled/default preflight preserves all existing JPG/WebP behavior and metadata merging.

### Step 2: Implement effective-format selection

Before the per-image loop, compute:

```python
requested_format = save_format or opts.samples_format
effective_format = scripts.scripts_postproc.output_extension(args, requested_format)
```

Make `output_exists()` accept an explicit extension. For each `PostprocessedImage`, prefer its recorded extension over the preflight extension and derive metadata/JPEG-quality handling from that final extension.

### Step 3: Run and commit

```powershell
.\venv\Scripts\python.exe -m pytest test\test_extras.py -q
git add modules/postprocessing.py test/test_extras.py
git commit -m "feat: preserve alpha in Extras PNG outputs"
```

## Task 8: Expose Background Removal through both Extras API endpoints

**Files:**

- Modify: `modules/api/models.py`
- Modify: `modules/postprocessing.py`
- Modify: `modules/api/api.py`
- Modify: `test/test_extras.py`

### Step 1: Write failing API tests

Add request-model and handler tests for:

- `background_removal_enabled` defaults to false.
- `background_removal_model` defaults to null.
- Both fields reach `run_extras()` and then the `Background Removal` script argument dictionary.
- Enabled requests with an unknown/unavailable model produce a client-visible 4xx response.
- Successful single and batch responses begin with the decoded PNG signature and preserve RGBA even when global `samples_format` is JPG.
- Disabled legacy Extras API requests retain their current serialization behavior.

### Step 2: Extend request and legacy Extras mapping

Add the two approved fields to `ExtrasBaseRequest`. Add optional parameters at the end of `run_extras()` and map them to:

```python
"Background Removal": {
    "enabled": background_removal_enabled,
    "model_name": background_removal_model,
}
```

### Step 3: Add an API encoding override

Change `encode_pil_to_base64()` to accept an optional output-format override while preserving every existing caller. Extras handlers pass `png` when background removal is enabled so the API serializer cannot flatten RGBA according to global settings.

Translate model-selection/contract errors into an appropriate FastAPI `HTTPException` with a concise detail string rather than returning a server traceback.

### Step 4: Run and commit

```powershell
.\venv\Scripts\python.exe -m pytest test\test_extras.py -q
git add modules/api/models.py modules/postprocessing.py modules/api/api.py test/test_extras.py
git commit -m "feat: expose background removal in Extras API"
```

## Task 9: Copy the supplied local models and add an opt-in smoke runner

**Files:**

- Local ignored assets: `models/bg-removal/*.onnx`
- Create: `test/background_removal_smoke.py`

### Step 1: Verify source and destination targets

Read the 13 source filenames from:

```text
C:\Users\rraze\OneDrive\Documents\Python Scripts\bg-removal\models
```

Confirm `models/**/*` is ignored and resolve the destination to `D:\stable-diffusion\v1.10.1\models\bg-removal` before copying.

### Step 2: Copy without overwriting mismatched existing files

Create the exact destination directory. For each source file:

- If absent, copy it.
- If present with the same SHA-256, keep it.
- If present with a different SHA-256, stop and report the conflict rather than overwrite.

After the copy, compare source/destination SHA-256 for all 13 files. Never stage the weights.

### Step 3: Add the smoke runner

Create a directly executable, non-pytest-collected script that:

- Accepts `--models-path`, `--image`, and optional repeated `--model` display names.
- Defaults to the configured model directory.
- Runs every installed registered model.
- Reports provider, elapsed time, output size/mode, and alpha min/max.
- Fails if output is not exact-size RGBA, alpha is non-finite, or alpha is constant.
- Does not save output unless an explicit output directory is supplied.

### Step 4: Run a limited real-model check first

Run the smallest model, then one model from each adapter family before attempting the complete set. On CPU, allow the complete smoke run to be an explicit long-running verification step.

Example:

```powershell
.\venv\Scripts\python.exe test\background_removal_smoke.py --image test\test_files\img2img_basic.png --model "RMBG 1.4"
```

### Step 5: Commit only the runner

```powershell
git add test/background_removal_smoke.py
git commit -m "test: add background removal model smoke runner"
```

## Task 10: Document installation, providers, API, and licenses

**Files:**

- Create: `docs/background-removal.md`
- Modify: `README.md` only if there is an existing Extras/features index appropriate for one link

### Step 1: Write the user documentation

Document:

- `models/bg-removal` and `--background-removal-models-path`.
- The table of exact filenames/display names.
- Installed-only discovery and no automatic downloads.
- Single, batch, directory, and API usage.
- The two API fields with JSON examples.
- Forced-PNG behavior.
- CPU baseline and how to replace it with exactly one compatible provider distribution.
- Provider fallback/logging.
- The real-model smoke command.
- Weight licensing responsibility, including RMBG 2.0's noncommercial default terms and separate commercial licensing.

Use links to primary model/provider documentation and avoid copying model-card text verbatim.

### Step 2: Check docs and commit

```powershell
rg -n "models/BackgroundRemoval|models\\BackgroundRemoval|TBD|TODO" docs/background-removal.md docs/superpowers/specs/2026-10-06-background-removal-design.md
git diff --check
git add docs/background-removal.md README.md
git commit -m "docs: explain Extras background removal"
```

If `README.md` has no suitable feature index, leave it unchanged and stage only the new document.

## Task 11: Complete regression and real-model verification

**Files:**

- Modify only files required to repair failures found by verification.

### Step 1: Run focused tests

```powershell
.\venv\Scripts\python.exe -m pytest test\test_background_removal_config.py test\test_background_removal.py test\test_background_removal_script.py test\test_scripts_postprocessing.py test\test_extras.py -q
```

Expected: all pass.

### Step 2: Run related Extras/API regression tests

```powershell
.\venv\Scripts\python.exe -m pytest test\test_api_image_inputs.py test\test_api_image_response.py test\test_extras.py -q
```

Expected: all pass; integration tests that require a running server may be separated and reported accurately.

### Step 3: Run static checks

```powershell
.\venv\Scripts\python.exe -m compileall modules\background_removal.py modules\scripts_postprocessing.py modules\postprocessing.py modules\api\models.py modules\api\api.py scripts\postprocessing_background_removal.py test\background_removal_smoke.py
git diff --check
git status --short
```

Expected: successful compilation, no whitespace errors, and no ONNX weights staged.

### Step 4: Run real-model smoke verification

Run at least one installed file from every adapter family. If resources permit, run all 13. Record exact successes, failures, providers, and any models not run; do not claim all-model compatibility from mocked tests alone.

### Step 5: Manual WebUI acceptance

Launch with the normal user configuration and verify:

1. Background Removal is last in Extras.
2. Only the copied 13 files appear.
3. Single Image produces a transparent PNG.
4. Uploaded batch reuses one session.
5. Directory batch plus Skip existing checks `.png` targets.
6. Upscale plus Background Removal upscales first and removes the background last.
7. A custom external path supplied through `--background-removal-models-path` replaces the default discovery root.
8. Both Extras endpoints accept the new fields and return PNG bytes.

### Step 6: Final review and integration decision

Use `superpowers:requesting-code-review`, fix validated findings through test-first changes, rerun the affected verification, and then use `superpowers:finishing-a-development-branch` to present merge/PR/keep options.
