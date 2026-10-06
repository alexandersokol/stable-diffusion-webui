import dataclasses
import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image


logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


class BackgroundRemovalError(RuntimeError):
    pass


class ModelContractError(BackgroundRemovalError):
    pass


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


@dataclasses.dataclass(frozen=True)
class InstalledModel:
    spec: ModelSpec
    path: Path


@dataclasses.dataclass(frozen=True)
class PreparedInput:
    tensor: np.ndarray
    source_size: tuple[int, int]
    target_size: tuple[int, int]
    content_box: tuple[int, int, int, int]


MODEL_SPECS = (
    ModelSpec("Anime Segmentation", "anime-seg.onnx", "isnet_anime", (1024, 1024), "img", "mask", "letterbox", "zero_to_one", "minmax"),
    ModelSpec("BEN2 Base", "BEN2_Base.onnx", "ben2_base", (1024, 1024), "input.1", "17728", "stretch", "zero_to_one", "minmax"),
    ModelSpec("BEN2 ONNX", "BEN2_ONNX.onnx", "ben2_imagenet", (1024, 1024), "pixel_values", "alphas", "stretch", "imagenet", "alpha"),
    ModelSpec("BiRefNet DIS", "BiRefNet-DIS.onnx", "birefnet", (1024, 1024), "input_image", "output_image", "stretch", "imagenet", "sigmoid_minmax"),
    ModelSpec("BiRefNet General", "BiRefNet-general.onnx", "birefnet", (1024, 1024), "input_image", "output_image", "stretch", "imagenet", "sigmoid_minmax"),
    ModelSpec("BiRefNet Massive", "BiRefNet-massive.onnx", "birefnet", (1024, 1024), "input_image", "output_image", "stretch", "imagenet", "sigmoid_minmax"),
    ModelSpec("BiRefNet Portrait", "BiRefNet-portrait.onnx", "birefnet", (1024, 1024), "input_image", "output_image", "stretch", "imagenet", "sigmoid_minmax"),
    ModelSpec("DeepLabV3 MobileViT Small", "deeplabv3-mobilevit-small.onnx", "deeplab_single_mask", (1024, 1024), "input", "output", "stretch", "imagenet", "alpha"),
    ModelSpec("ISNet Anime", "isnet-anime.onnx", "isnet_anime", (1024, 1024), "img", "mask", "letterbox", "zero_to_one", "minmax"),
    ModelSpec("RMBG 1.4", "RMBG-1.4.onnx", "rmbg_1_4", (1024, 1024), "pixel_values", "logits", "stretch", "imagenet", "sigmoid_minmax"),
    ModelSpec("RMBG 2.0", "RMBG-2.0.onnx", "rmbg_2_0", (1024, 1024), "pixel_values", "alphas", "stretch", "imagenet", "alpha"),
    ModelSpec("Silueta", "silueta.onnx", "u2net", (320, 320), "input.1", "1959", "stretch", "imagenet", "minmax"),
    ModelSpec("U2Net Human Segmentation", "u2net_human_seg.onnx", "u2net", (320, 320), "input.1", "1959", "stretch", "imagenet", "minmax"),
)


def discover_models(root: str | os.PathLike[str]) -> dict[str, InstalledModel]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        return {}

    registered_filenames = {spec.filename for spec in MODEL_SPECS}
    candidates: dict[str, list[Path]] = {filename: [] for filename in registered_filenames}

    for path in root_path.rglob("*"):
        if path.is_file() and path.name in registered_filenames:
            candidates[path.name].append(path.resolve())

    installed = {}
    for spec in MODEL_SPECS:
        paths = sorted(candidates[spec.filename], key=lambda path: (str(path).casefold(), str(path)))
        if not paths:
            continue

        selected = paths[0]
        if len(paths) > 1:
            logger.warning(
                "Multiple files named %s were found; using %s and ignoring: %s",
                spec.filename,
                selected,
                ", ".join(str(path) for path in paths[1:]),
            )

        installed[spec.display_name] = InstalledModel(spec=spec, path=selected)

    return installed


def prepare_input(image: Image.Image, spec: ModelSpec) -> PreparedInput:
    source = image.convert("RGB")
    source_width, source_height = source.size
    target_width, target_height = spec.input_size

    if source_width < 1 or source_height < 1 or target_width < 1 or target_height < 1:
        raise ModelContractError(f"{spec.display_name} requires non-empty source and input dimensions")

    if spec.resize_mode == "stretch":
        resized = source.resize((target_width, target_height), Image.Resampling.LANCZOS)
        content_box = (0, 0, target_width, target_height)
    elif spec.resize_mode == "letterbox":
        scale = min(target_width / source_width, target_height / source_height)
        resized_width = max(1, min(target_width, round(source_width * scale)))
        resized_height = max(1, min(target_height, round(source_height * scale)))
        foreground = source.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
        left = (target_width - resized_width) // 2
        top = (target_height - resized_height) // 2
        resized = Image.new("RGB", (target_width, target_height))
        resized.paste(foreground, (left, top))
        content_box = (left, top, left + resized_width, top + resized_height)
    else:
        raise ModelContractError(f"{spec.display_name} has unsupported resize mode {spec.resize_mode!r}")

    array = np.asarray(resized, dtype=np.float32) / np.float32(255.0)
    if spec.normalization == "imagenet":
        array = (array - IMAGENET_MEAN) / IMAGENET_STD
    elif spec.normalization != "zero_to_one":
        raise ModelContractError(f"{spec.display_name} has unsupported normalization {spec.normalization!r}")

    tensor = np.ascontiguousarray(array.transpose(2, 0, 1)[None, ...], dtype=np.float32)
    return PreparedInput(
        tensor=tensor,
        source_size=source.size,
        target_size=(target_width, target_height),
        content_box=content_box,
    )


def _single_mask(output, spec: ModelSpec) -> np.ndarray:
    mask = np.asarray(output, dtype=np.float32)
    if mask.ndim == 4:
        if mask.shape[0] != 1 or mask.shape[1] != 1:
            raise ModelContractError(f"{spec.display_name} output must contain a single foreground mask")
        mask = mask[0, 0]
    elif mask.ndim == 3:
        if mask.shape[0] != 1:
            raise ModelContractError(f"{spec.display_name} output must contain a single foreground mask")
        mask = mask[0]
    elif mask.ndim != 2:
        raise ModelContractError(f"{spec.display_name} output must contain a single foreground mask")

    if mask.shape[0] < 1 or mask.shape[1] < 1:
        raise ModelContractError(f"{spec.display_name} output mask is empty")
    if not np.isfinite(mask).all():
        raise ModelContractError(f"{spec.display_name} output mask must contain only finite values")

    return mask


def _activate_mask(mask: np.ndarray, spec: ModelSpec) -> np.ndarray:
    if spec.activation == "alpha":
        return np.clip(mask, 0.0, 1.0)

    if spec.activation == "sigmoid_minmax":
        mask = 1.0 / (1.0 + np.exp(-np.clip(mask, -80.0, 80.0)))
    elif spec.activation != "minmax":
        raise ModelContractError(f"{spec.display_name} has unsupported activation {spec.activation!r}")

    minimum = float(mask.min())
    maximum = float(mask.max())
    if maximum - minimum <= np.finfo(np.float32).eps:
        raise ModelContractError(f"{spec.display_name} returned a constant mask that cannot be normalized")

    return (mask - minimum) / (maximum - minimum)


def reconstruct_mask(outputs: dict[str, np.ndarray], prepared: PreparedInput, spec: ModelSpec) -> Image.Image:
    if spec.output_name not in outputs:
        available = ", ".join(sorted(outputs)) or "none"
        raise ModelContractError(
            f"{spec.display_name} did not return required output {spec.output_name!r}; available outputs: {available}"
        )

    mask = _activate_mask(_single_mask(outputs[spec.output_name], spec), spec)
    if not np.isfinite(mask).all():
        raise ModelContractError(f"{spec.display_name} produced a non-finite mask after activation")

    target_width, target_height = prepared.target_size
    mask_image = Image.fromarray(mask.astype(np.float32), mode="F")
    if mask_image.size != prepared.target_size:
        mask_image = mask_image.resize((target_width, target_height), Image.Resampling.BILINEAR)

    if prepared.content_box != (0, 0, target_width, target_height):
        mask_image = mask_image.crop(prepared.content_box)

    if mask_image.size != prepared.source_size:
        mask_image = mask_image.resize(prepared.source_size, Image.Resampling.BILINEAR)

    mask_array = np.asarray(mask_image, dtype=np.float32)
    mask_bytes = np.rint(np.clip(mask_array, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(mask_bytes, mode="L")


def compose_rgba(image: Image.Image, mask: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    if mask.mode != "L":
        mask = mask.convert("L")
    if mask.size != rgb.size:
        raise ModelContractError(f"Alpha mask size {mask.size} does not match image size {rgb.size}")

    red, green, blue = rgb.split()
    return Image.merge("RGBA", (red, green, blue, mask))
