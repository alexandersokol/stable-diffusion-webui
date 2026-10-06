import dataclasses
import logging
import os
from pathlib import Path


logger = logging.getLogger(__name__)


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
