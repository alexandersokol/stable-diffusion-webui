# Extras Background Removal Design

## Goal

Add portable, model-selectable background removal to the Extras tab. The operation must support all 13 supplied ONNX models, reuse every existing Extras input mode, run after every other enabled postprocessor, and preserve transparency by saving affected outputs as PNG.

## Scope

This change includes:

- An optional `Background Removal` postprocessing section in Extras.
- A registry and multi-architecture adapter subsystem for the 13 supplied ONNX files.
- Local model discovery with a configurable model directory.
- Single-image, uploaded-batch, directory-batch, and REST API support.
- Portable ONNX Runtime execution with an accelerated-provider preference and CPU fallback.
- Per-run PNG output selection when background removal is enabled.
- Tests, documentation, and a local all-model smoke test.

This change does not include model downloads, remote model catalogs, training, mask editing, foreground replacement, or generic threshold/matting controls.

## Model Directory and Discovery

The default directory is:

```text
models/bg-removal
```

A new command-line option, `--background-removal-models-path`, overrides that directory. Its path handling should follow existing model-directory options, including normalization and support for a location outside the WebUI checkout.

Discovery is recursive and limited to exact filenames in the registry. Only entries whose files exist are shown in the UI or accepted by the API. Discovery never downloads a model or substitutes a different model. If duplicate exact filenames exist below the configured root, deterministic path ordering selects one and emits a warning identifying the selected path.

The supplied weights will be copied from the original project into the default directory for local use during implementation. They remain untracked and are not added to release packages or Git history.

## Supported Models

The registry contains a stable display name, exact filename, adapter family, input contract, and output contract for each supplied model:

| Display name | Filename | Adapter family |
| --- | --- | --- |
| Anime Segmentation | `anime-seg.onnx` | ISNet/anime |
| BEN2 Base | `BEN2_Base.onnx` | BEN2 base |
| BEN2 ONNX | `BEN2_ONNX.onnx` | BEN2 ImageNet |
| BiRefNet DIS | `BiRefNet-DIS.onnx` | BiRefNet |
| BiRefNet General | `BiRefNet-general.onnx` | BiRefNet |
| BiRefNet Massive | `BiRefNet-massive.onnx` | BiRefNet |
| BiRefNet Portrait | `BiRefNet-portrait.onnx` | BiRefNet |
| DeepLabV3 MobileViT Small | `deeplabv3-mobilevit-small.onnx` | DeepLab single-mask |
| ISNet Anime | `isnet-anime.onnx` | ISNet/anime |
| RMBG 1.4 | `RMBG-1.4.onnx` | RMBG 1.4 |
| RMBG 2.0 | `RMBG-2.0.onnx` | RMBG 2.0 |
| Silueta | `silueta.onnx` | U2Net/Silueta |
| U2Net Human Segmentation | `u2net_human_seg.onnx` | U2Net/Silueta |

`anime-seg.onnx` and `isnet-anime.onnx` are separate registry entries even when the supplied files are byte-identical. Presence is evaluated separately for each filename.

Architecture-specific input sizes, channel normalization, letterboxing or stretching, input/output names, sigmoid/min-max processing, and restoration to source dimensions are properties of the adapter or model specification rather than user-facing settings.

## Components

### Registry

An immutable registry describes all supported files and maps them to adapter implementations. UI choices and API validation use the same discovered-model view so they cannot drift apart.

### Adapters

Adapters implement:

1. Conversion of a Pillow RGB image into the model's input tensor.
2. ONNX input and output contract validation.
3. Conversion of the selected output tensor into a finite single-channel alpha mask.
4. Cropping or resizing the mask back to the exact source dimensions.

Pillow and NumPy provide image conversion, resizing, normalization, and RGBA composition. The subsystem does not depend on `rembg`, OpenCV, TorchVision, SciPy, or a second deep-learning framework.

The original source RGB values are preserved. Only the alpha channel is derived from the predicted mask.

### Runtime Service

A background-removal service owns ONNX Runtime session creation and inference. It caches one active session at a time, which is reused for a batch. Selecting or replacing a model releases the previous session so several large networks are not retained simultaneously.

The cache identity includes the resolved path, file size, modification time, and provider selection. A lock serializes session creation, cache changes, and inference to prevent races between UI and API requests.

### Extras Script

A postprocessing script owns the UI arguments and invokes the service. It is disabled by default and does no work unless enabled.

The postprocessing framework gains a general terminal-operation capability. Background Removal marks itself terminal, so it sorts after all ordinary operations regardless of the configurable postprocessing order. Existing non-terminal scripts retain their current order.

The framework also gains a preflight output-extension hook. Background Removal reports `png` when enabled, allowing skip-existing checks to use the effective destination before expensive processing begins. The saved postprocessed image records the same preferred extension as a safety invariant.

## User Interface

The Extras tab shows a `Background Removal` accordion after the existing postprocessing sections. It is disabled by default and contains:

- An enable/disable control supplied by the existing input-accordion pattern.
- A model dropdown populated from installed registry entries.
- A concise note that enabled runs are saved as PNG to preserve transparency.

The section reuses the existing Single Image, Batch Process, and Batch from Directory inputs and output controls. No duplicate upload or directory controls are added.

If no registered model exists, the section remains visible but cannot be enabled. It explains that models must be placed below the resolved background-removal model directory. If a selected file disappears after the UI was built, execution fails with a clear model-and-path error.

The initial version intentionally exposes no generic mask threshold, erosion, feathering, or alpha-matting controls. The supplied architectures do not share equivalent semantics for such settings.

## Processing and Output Flow

For every input image:

1. Existing enabled non-terminal Extras operations run in their configured order.
2. Background Removal receives the result of those operations.
3. The selected adapter generates an alpha mask at the postprocessed image's current dimensions.
4. The service combines the unchanged RGB pixels with the mask and returns RGBA.
5. The output is encoded and, when requested, saved as PNG.

Enabling Background Removal overrides JPG or WebP output selection for affected images because those paths currently discard alpha. When the feature is disabled, save-format selection and all existing Extras behavior remain unchanged.

Skip-existing logic evaluates the effective `.png` output path before model inference. Naming, metadata preservation, output-directory behavior, and generated response structure otherwise follow the existing Extras pipeline.

## REST API

The existing Extras request model gains:

- `background_removal_enabled: bool = false`
- `background_removal_model: str | null = null`

These fields apply to both single-image and batch Extras endpoints. When disabled, they preserve current behavior. When enabled, the model value must resolve to an installed registry entry. Missing, unknown, or unavailable selections return a clear client-visible validation error rather than choosing the first model silently.

Successful enabled responses contain an RGBA PNG image encoded through the existing response mechanism. Disk saves, when requested by the endpoint, use the same forced-PNG behavior as the UI.

## ONNX Runtime and Portability

CPU ONNX Runtime is the portable baseline. The runtime inspects the providers available in the installed ONNX Runtime package and prefers an applicable accelerated provider, retaining CPU as the final fallback. Provider selection must not assume Windows or NVIDIA hardware.

Optional provider-capable installations may expose CUDA, DirectML, CoreML, or other supported providers without a change to feature code. The project must avoid installing conflicting ONNX Runtime distributions together.

Session creation first attempts the selected provider list. If provider initialization fails, it logs the reason and retries once with CPU. An inference or model-contract failure is not treated as a reason to switch architecture or model.

One informational log entry per new session records the display name, resolved file, and providers actually in use.

## Validation and Failure Handling

The subsystem fails explicitly when:

- A selected file is missing or unreadable.
- An ONNX graph cannot be loaded.
- Required input or output tensors are absent or incompatible.
- Output dimensions cannot be interpreted as a single foreground mask.
- A result contains NaN or infinite values.
- Mask normalization cannot produce a meaningful finite mask.

Errors include the selected model and resolved path where applicable. The system never saves the unchanged input as though background removal succeeded and never silently switches to another installed model.

## Licensing

The implementation documentation states that model weights retain their own licenses and are not licensed by the WebUI feature. RMBG 2.0 requires an explicit note about its noncommercial terms and separate commercial licensing. The UI and documentation must not imply that every locally supplied model is unrestricted.

## Testing

Fast automated tests use temporary model files and mocked ONNX sessions. They cover:

- Exact discovery for each of the 13 filenames.
- Missing files, recursively located files, deterministic duplicate handling, and empty directories.
- Separate registry visibility for the byte-identical anime filenames.
- Preprocessing shapes and normalization for each adapter family.
- Output selection, sigmoid/min-max processing, finite-value validation, source-size restoration, and RGBA composition.
- One-session caching, reuse through a batch, invalidation after file replacement, provider fallback, and locking.
- Background Removal sorting after Upscale and face restoration even when user order settings conflict.
- Forced PNG preflight, save behavior, and skip-existing checks.
- Single-image, uploaded-batch, and directory-batch flows.
- UI state with zero, one, and multiple available models.
- API defaults, successful selection, missing/unknown selections, and RGBA PNG responses.
- Regression behavior for JPG/WebP output and existing Extras operations when disabled.

A separate opt-in smoke test or verification script runs every installed real model against a small reference image. It checks successful inference, exact output dimensions, RGBA mode, finite alpha values, and a non-empty/non-constant mask. This test is excluded from the normal suite because the complete model set is approximately 4.5 GiB and inference cost varies significantly by provider.

## Documentation

User-facing documentation covers:

- The default `models/bg-removal` location.
- The `--background-removal-models-path` override.
- All supported exact filenames and display names.
- The no-download and installed-only discovery rules.
- Provider selection and CPU fallback.
- UI and API usage.
- Forced-PNG behavior.
- Model-license responsibility and the RMBG 2.0 restriction.
- Running the optional all-model smoke test.

## Acceptance Criteria

The feature is complete when all 13 supplied files can be independently discovered and processed through their correct adapters; only installed files appear; every Extras input mode and both Extras API endpoints work; Background Removal always runs last; enabled outputs preserve transparency as PNG; the model path is configurable; CPU execution remains portable; existing Extras behavior is unchanged when disabled; and focused plus regression tests pass.
