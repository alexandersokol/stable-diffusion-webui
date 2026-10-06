# Extras background removal

Background Removal is an optional final operation in the **Extras** tab. It uses a locally installed ONNX model, preserves the input dimensions, and returns an RGBA image whose alpha channel is the predicted foreground mask.

## Install models

The default model root is:

```text
models/bg-removal
```

Discovery is recursive, but filenames are exact and case-sensitive. Only registered files that exist below the configured root are shown in the UI; the WebUI does not download weights. Restart the WebUI after adding, removing, or renaming model files so the Extras choices are rebuilt.

| UI name | Exact filename | Model family/reference |
| --- | --- | --- |
| Anime Segmentation | `anime-seg.onnx` | [skytnt/anime-seg](https://huggingface.co/skytnt/anime-seg) |
| BEN2 Base | `BEN2_Base.onnx` | [PramaLLC/BEN2](https://huggingface.co/PramaLLC/BEN2) |
| BEN2 ONNX | `BEN2_ONNX.onnx` | [onnx-community/BEN2-ONNX](https://huggingface.co/onnx-community/BEN2-ONNX) |
| BiRefNet DIS | `BiRefNet-DIS.onnx` | [ZhengPeng7/BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) |
| BiRefNet General | `BiRefNet-general.onnx` | [ZhengPeng7/BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) |
| BiRefNet Massive | `BiRefNet-massive.onnx` | [ZhengPeng7/BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) |
| BiRefNet Portrait | `BiRefNet-portrait.onnx` | [ZhengPeng7/BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) |
| DeepLabV3 MobileViT Small | `deeplabv3-mobilevit-small.onnx` | DeepLabV3/MobileViT semantic segmentation |
| ISNet Anime | `isnet-anime.onnx` | [rembg model catalog](https://github.com/danielgatis/rembg#models) |
| RMBG 1.4 | `RMBG-1.4.onnx` | Supplied 21-class DeepLab/MobileViT export; it is not interchangeable with BRIA's single-mask export |
| RMBG 2.0 | `RMBG-2.0.onnx` | [briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) |
| Silueta | `silueta.onnx` | [rembg model catalog](https://github.com/danielgatis/rembg#models) |
| U2Net Human Segmentation | `u2net_human_seg.onnx` | [rembg model catalog](https://github.com/danielgatis/rembg#models) |

When duplicate exact filenames exist in subdirectories, deterministic path ordering chooses one and the log identifies the selected and ignored paths.

To keep the weights elsewhere, pass an absolute or portable relative path when starting the WebUI:

```powershell
webui-user.bat --background-removal-models-path "D:\AI models\bg-removal"
```

The same option can be added to `COMMANDLINE_ARGS` in `webui-user.bat`.

## Extras UI

Open **Extras**, choose any of its existing input modes, then enable **Background Removal** and select an installed model:

- **Single Image** accepts one pasted, selected, or drag-and-dropped image.
- **Batch Process** accepts multiple uploaded images.
- **Batch from Directory** uses the existing input and output directory fields.

Other Extras operations can remain enabled. Background Removal is terminal and always runs after face restoration, upscaling, and every other non-terminal postprocessor, regardless of the visible script order.

An enabled background-removal run is always encoded and saved as PNG, even if the configured Extras format is JPEG or WebP. This is required to retain transparency. Disabled runs keep the existing format behavior. Existing naming, metadata, output-directory, and skip-existing rules still apply; skip-existing checks the effective `.png` path before inference.

## REST API

Both existing Extras endpoints accept two additional fields:

```json
{
  "background_removal_enabled": true,
  "background_removal_model": "BiRefNet General"
}
```

`background_removal_enabled` defaults to `false`. When it is `true`, `background_removal_model` must exactly match an installed UI name from the table. Missing, unknown, or currently unavailable selections return HTTP 422 rather than selecting another model.

Single-image request:

```json
POST /sdapi/v1/extra-single-image
{
  "image": "<base64 image bytes>",
  "background_removal_enabled": true,
  "background_removal_model": "RMBG 2.0",
  "show_extras_results": true
}
```

Batch request:

```json
POST /sdapi/v1/extra-batch-images
{
  "imageList": [
    {"name": "subject.png", "data": "<base64 image bytes>"}
  ],
  "background_removal_enabled": true,
  "background_removal_model": "Silueta",
  "show_extras_results": true
}
```

The response image strings contain base64-encoded PNG bytes when background removal is enabled.

## ONNX Runtime providers

The pinned `onnxruntime` dependency is the portable CPU baseline. The loader asks ONNX Runtime for providers actually present in the environment, prefers an accelerator, keeps CPU as a fallback, and logs the providers that loaded each model. A provider load failure is retried with `CPUExecutionProvider`; inference or model-contract failures include the display name and resolved file path.

ONNX Runtime documents the available [execution providers](https://onnxruntime.ai/docs/execution-providers/) and [Python installation matrix](https://onnxruntime.ai/docs/install/). Install exactly one compatible ONNX Runtime distribution in the WebUI Python environment. For example, replace the CPU package with one matching the pinned version:

```powershell
venv\Scripts\python.exe -m pip uninstall -y onnxruntime
venv\Scripts\python.exe -m pip install onnxruntime-gpu==1.21.0
```

Use the distribution appropriate for the machine, such as `onnxruntime-gpu`, `onnxruntime-directml`, or `onnxruntime-openvino`; do not install multiple ONNX Runtime distributions together. GPU packages also require compatible system drivers and, where applicable, CUDA/cuDNN libraries. If those native dependencies are unavailable, use the CPU baseline.

## Real-model smoke test

The opt-in runner does not download weights and does not save images unless `--output-dir` is supplied. With the WebUI environment active:

```powershell
venv\Scripts\python.exe test\background_removal_smoke.py `
  --models-path models\bg-removal `
  --image test\test_files\img2img_basic.png
```

Omit `--model` to test every installed registered file, or repeat it to select models:

```powershell
venv\Scripts\python.exe test\background_removal_smoke.py `
  --models-path models\bg-removal `
  --image test\test_files\img2img_basic.png `
  --model "BEN2 Base" `
  --model "RMBG 2.0" `
  --output-dir tmp\background-removal-smoke
```

Each result must be RGBA, match the source dimensions, and contain a finite, non-constant alpha channel. The runner reports the active provider and inference time.

## Weight licenses

Model weights are not part of the WebUI source distribution. You are responsible for obtaining each file lawfully and complying with its model card, license, attribution, redistribution, and use restrictions. Similar filenames or ONNX conversions do not guarantee identical terms; verify the exact artifact you install.

In particular, BRIA publishes the self-hosted RMBG 2.0 weights under CC BY-NC 4.0 for noncommercial use. Commercial use requires a separate agreement from BRIA; see the [RMBG 2.0 model card](https://huggingface.co/briaai/RMBG-2.0) for the current terms. This documentation is operational guidance, not legal advice.
