import dataclasses
import importlib
import logging
import os
import threading
from pathlib import Path

import numpy as np
from PIL import Image


logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
PROVIDER_PREFERENCE = (
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "DmlExecutionProvider",
    "CoreMLExecutionProvider",
    "OpenVINOExecutionProvider",
    "CPUExecutionProvider",
)


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


def _load_onnxruntime():
    try:
        return importlib.import_module("onnxruntime")
    except ImportError as exc:
        raise BackgroundRemovalError(
            "Background removal requires ONNX Runtime; install exactly one compatible onnxruntime distribution"
        ) from exc


class BackgroundRemovalService:
    def __init__(self, runtime_loader=None):
        self._runtime_loader = runtime_loader or _load_onnxruntime
        self._runtime = None
        self._session = None
        self._session_identity = None
        self._lock = threading.RLock()

    def _get_runtime(self):
        if self._runtime is None:
            self._runtime = self._runtime_loader()
        return self._runtime

    @staticmethod
    def _providers(runtime) -> list[str]:
        available = set(runtime.get_available_providers())
        providers = [name for name in PROVIDER_PREFERENCE if name in available]
        if "CPUExecutionProvider" not in providers:
            providers.append("CPUExecutionProvider")
        return providers

    @staticmethod
    def _model_error(model: InstalledModel, message: str) -> ModelContractError:
        return ModelContractError(f"{model.spec.display_name} ({model.path.resolve()}): {message}")

    def _validate_session(self, session, model: InstalledModel) -> None:
        spec = model.spec
        inputs = {value.name: value for value in session.get_inputs()}
        if spec.input_name not in inputs:
            raise self._model_error(model, f"required input {spec.input_name!r} is missing")

        shape = list(inputs[spec.input_name].shape)
        expected = [1, 3, spec.input_size[1], spec.input_size[0]]
        if len(shape) != 4:
            raise self._model_error(model, f"input shape {shape!r} is not four-dimensional NCHW")
        for actual, wanted in zip(shape, expected):
            if isinstance(actual, int) and actual != wanted:
                raise self._model_error(model, f"input shape {shape!r} does not match expected {expected!r}")

        output_names = {value.name for value in session.get_outputs()}
        if spec.output_name not in output_names:
            raise self._model_error(model, f"required output {spec.output_name!r} is missing")

    def _create_session(self, runtime, model: InstalledModel, providers: list[str]):
        path = str(model.path.resolve())
        try:
            session = runtime.InferenceSession(path, providers=providers)
        except Exception as exc:
            if providers == ["CPUExecutionProvider"]:
                raise BackgroundRemovalError(
                    f"Could not load {model.spec.display_name} ({path}) with CPUExecutionProvider: {exc}"
                ) from exc

            logger.warning(
                "Could not load %s with providers %s (%s); retrying with CPU",
                model.spec.display_name,
                providers,
                exc,
            )
            try:
                session = runtime.InferenceSession(path, providers=["CPUExecutionProvider"])
            except Exception as cpu_exc:
                raise BackgroundRemovalError(
                    f"Could not load {model.spec.display_name} ({path}) with CPUExecutionProvider: {cpu_exc}"
                ) from cpu_exc

        self._validate_session(session, model)
        logger.info(
            "Loaded background-removal model %s from %s with providers %s",
            model.spec.display_name,
            path,
            session.get_providers(),
        )
        return session

    def _get_session(self, model: InstalledModel):
        path = model.path.resolve()
        try:
            stat = path.stat()
        except OSError as exc:
            raise BackgroundRemovalError(f"Background-removal model {model.spec.display_name} is unavailable at {path}: {exc}") from exc

        runtime = self._get_runtime()
        providers = self._providers(runtime)
        identity = (str(path), stat.st_size, stat.st_mtime_ns, tuple(providers))
        if self._session is not None and self._session_identity == identity:
            return self._session

        if self._session is not None:
            previous = self._session
            self._session = None
            self._session_identity = None
            del previous

        session = self._create_session(runtime, model, providers)
        self._session = session
        self._session_identity = identity
        return session

    def remove_background(self, image: Image.Image, model: InstalledModel) -> Image.Image:
        prepared = prepare_input(image, model.spec)
        with self._lock:
            session = self._get_session(model)
            try:
                values = session.run([model.spec.output_name], {model.spec.input_name: prepared.tensor})
            except Exception as exc:
                raise BackgroundRemovalError(
                    f"Inference failed for {model.spec.display_name} ({model.path.resolve()}): {exc}"
                ) from exc

            if len(values) != 1:
                raise self._model_error(model, f"expected one output value, received {len(values)}")

            try:
                mask = reconstruct_mask({model.spec.output_name: values[0]}, prepared, model.spec)
            except ModelContractError as exc:
                raise self._model_error(model, str(exc)) from exc

            return compose_rgba(image, mask)

    def unload(self) -> None:
        with self._lock:
            if self._session is not None:
                previous = self._session
                self._session = None
                self._session_identity = None
                del previous
