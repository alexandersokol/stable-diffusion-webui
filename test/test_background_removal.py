import dataclasses
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from modules import background_removal


EXPECTED_FILENAMES = {
    "anime-seg.onnx",
    "BEN2_Base.onnx",
    "BEN2_ONNX.onnx",
    "BiRefNet-DIS.onnx",
    "BiRefNet-general.onnx",
    "BiRefNet-massive.onnx",
    "BiRefNet-portrait.onnx",
    "deeplabv3-mobilevit-small.onnx",
    "isnet-anime.onnx",
    "RMBG-1.4.onnx",
    "RMBG-2.0.onnx",
    "silueta.onnx",
    "u2net_human_seg.onnx",
}


def test_registry_contains_the_exact_supported_model_files():
    specs = background_removal.MODEL_SPECS

    assert isinstance(specs, tuple)
    assert {spec.filename for spec in specs} == EXPECTED_FILENAMES
    assert len({spec.display_name for spec in specs}) == 13
    assert all(dataclasses.is_dataclass(spec) for spec in specs)
    assert all(spec.__dataclass_params__.frozen for spec in specs)


def test_registry_records_architecture_contracts():
    specs = {spec.filename: spec for spec in background_removal.MODEL_SPECS}

    assert specs["anime-seg.onnx"].adapter == "isnet_anime"
    assert specs["anime-seg.onnx"].resize_mode == "letterbox"
    assert specs["BEN2_Base.onnx"].normalization == "zero_to_one"
    assert specs["BEN2_ONNX.onnx"].normalization == "imagenet"
    assert specs["BiRefNet-general.onnx"].activation == "sigmoid_minmax"
    assert specs["RMBG-2.0.onnx"].output_name == "alphas"
    assert specs["silueta.onnx"].input_size == (320, 320)
    assert specs["u2net_human_seg.onnx"].input_size == (320, 320)
    assert specs["deeplabv3-mobilevit-small.onnx"].input_size == (1024, 1024)


def test_discovery_returns_empty_without_creating_the_root(tmp_path):
    root = tmp_path / "missing-model-root"

    assert background_removal.discover_models(root) == {}
    assert not root.exists()


def test_discovery_includes_only_installed_registered_files(tmp_path):
    nested = tmp_path / "portraits"
    nested.mkdir()
    registered = nested / "BiRefNet-portrait.onnx"
    registered.write_bytes(b"portrait")
    (tmp_path / "unknown.onnx").write_bytes(b"unknown")

    installed = background_removal.discover_models(tmp_path)

    assert list(installed) == ["BiRefNet Portrait"]
    assert installed["BiRefNet Portrait"].path == registered.resolve()
    assert installed["BiRefNet Portrait"].spec.filename == registered.name


def test_discovery_keeps_both_named_anime_entries(tmp_path):
    (tmp_path / "anime-seg.onnx").write_bytes(b"same graph")
    (tmp_path / "isnet-anime.onnx").write_bytes(b"same graph")

    installed = background_removal.discover_models(tmp_path)

    assert list(installed) == ["Anime Segmentation", "ISNet Anime"]


def test_discovery_selects_a_deterministic_duplicate_and_warns(tmp_path, caplog):
    first_dir = tmp_path / "a"
    second_dir = tmp_path / "z"
    first_dir.mkdir()
    second_dir.mkdir()
    expected = first_dir / "RMBG-1.4.onnx"
    expected.write_bytes(b"first")
    (second_dir / "RMBG-1.4.onnx").write_bytes(b"second")

    with caplog.at_level(logging.WARNING, logger="modules.background_removal"):
        installed = background_removal.discover_models(tmp_path)

    assert installed["RMBG 1.4"].path == expected.resolve()
    assert "RMBG-1.4.onnx" in caplog.text
    assert str(expected.resolve()) in caplog.text


def adapter_spec(**overrides):
    values = {
        "display_name": "Test model",
        "filename": "test.onnx",
        "adapter": "test",
        "input_size": (2, 2),
        "input_name": "input",
        "output_name": "output",
        "resize_mode": "stretch",
        "normalization": "zero_to_one",
        "activation": "alpha",
    }
    values.update(overrides)
    return background_removal.ModelSpec(**values)


def test_prepare_input_converts_rgb_to_nchw_float32_zero_to_one():
    image = Image.new("RGB", (2, 2))
    image.putdata([
        (0, 127, 255),
        (255, 0, 127),
        (127, 255, 0),
        (64, 128, 192),
    ])

    prepared = background_removal.prepare_input(image, adapter_spec())

    assert prepared.tensor.shape == (1, 3, 2, 2)
    assert prepared.tensor.dtype == np.float32
    np.testing.assert_allclose(prepared.tensor[0, :, 0, 0], [0.0, 127 / 255, 1.0])
    assert prepared.source_size == (2, 2)
    assert prepared.content_box == (0, 0, 2, 2)


def test_prepare_input_applies_imagenet_normalization():
    image = Image.new("RGB", (1, 1), (255, 128, 0))
    spec = adapter_spec(input_size=(1, 1), normalization="imagenet")

    prepared = background_removal.prepare_input(image, spec)

    expected = [
        (1.0 - 0.485) / 0.229,
        ((128 / 255) - 0.456) / 0.224,
        (0.0 - 0.406) / 0.225,
    ]
    np.testing.assert_allclose(prepared.tensor[0, :, 0, 0], expected, rtol=1e-6)


def test_letterbox_mask_reconstruction_removes_padding():
    image = Image.new("RGB", (4, 2), "white")
    spec = adapter_spec(input_size=(8, 8), resize_mode="letterbox")
    prepared = background_removal.prepare_input(image, spec)
    output = np.zeros((1, 1, 8, 8), dtype=np.float32)
    output[:, :, 2:6, :] = 1.0

    mask = background_removal.reconstruct_mask({"output": output}, prepared, spec)

    assert prepared.content_box == (0, 2, 8, 6)
    assert mask.mode == "L"
    assert mask.size == (4, 2)
    assert set(mask.getdata()) == {255}


@pytest.mark.parametrize(
    "output",
    [
        np.array([[[[0.0, 1.0], [0.5, 0.25]]]], dtype=np.float32),
        np.array([[[0.0, 1.0], [0.5, 0.25]]], dtype=np.float32),
        np.array([[0.0, 1.0], [0.5, 0.25]], dtype=np.float32),
    ],
)
def test_mask_reconstruction_accepts_single_mask_shapes(output):
    prepared = background_removal.prepare_input(Image.new("RGB", (2, 2)), adapter_spec())

    mask = background_removal.reconstruct_mask({"output": output}, prepared, adapter_spec())

    assert list(mask.getdata()) == [0, 255, 128, 64]


def test_mask_reconstruction_applies_sigmoid_and_minmax():
    spec = adapter_spec(activation="sigmoid_minmax")
    prepared = background_removal.prepare_input(Image.new("RGB", (2, 2)), spec)
    logits = np.array([[-2.0, 0.0], [2.0, 1.0]], dtype=np.float32)

    mask = background_removal.reconstruct_mask({"output": logits}, prepared, spec)

    values = list(mask.getdata())
    assert values[0] == 0
    assert values[2] == 255
    assert values[0] < values[1] < values[3] < values[2]


@pytest.mark.parametrize(
    ("outputs", "activation", "message"),
    [
        ({"wrong": np.zeros((1, 1, 2, 2), dtype=np.float32)}, "alpha", "output"),
        ({"output": np.zeros((1, 2, 2, 2), dtype=np.float32)}, "alpha", "single foreground mask"),
        ({"output": np.array([[0.0, np.nan], [0.5, 1.0]], dtype=np.float32)}, "alpha", "finite"),
        ({"output": np.array([[0.0, np.inf], [0.5, 1.0]], dtype=np.float32)}, "alpha", "finite"),
        ({"output": np.ones((2, 2), dtype=np.float32)}, "minmax", "constant"),
    ],
)
def test_mask_reconstruction_rejects_invalid_outputs(outputs, activation, message):
    spec = adapter_spec(activation=activation)
    prepared = background_removal.prepare_input(Image.new("RGB", (2, 2)), spec)

    with pytest.raises(background_removal.ModelContractError, match=message):
        background_removal.reconstruct_mask(outputs, prepared, spec)


def test_compose_rgba_preserves_rgb_pixels_and_uses_mask_as_alpha():
    image = Image.new("RGB", (2, 1))
    image.putdata([(10, 20, 30), (40, 50, 60)])
    mask = Image.new("L", (2, 1))
    mask.putdata([0, 200])

    result = background_removal.compose_rgba(image, mask)

    assert result.mode == "RGBA"
    assert list(result.getdata()) == [(10, 20, 30, 0), (40, 50, 60, 200)]


class FakeSession:
    def __init__(self, spec, providers, output=None, input_name=None, input_shape=None, output_name=None):
        self.spec = spec
        self.providers = providers
        self.output = output if output is not None else np.array([[[[0.0, 1.0], [0.25, 0.75]]]], dtype=np.float32)
        self.input_name = input_name or spec.input_name
        self.input_shape = input_shape or [1, 3, spec.input_size[1], spec.input_size[0]]
        self.output_name = output_name or spec.output_name
        self.run_count = 0

    def get_inputs(self):
        return [SimpleNamespace(name=self.input_name, shape=self.input_shape)]

    def get_outputs(self):
        return [SimpleNamespace(name=self.output_name, shape=[1, 1, self.spec.input_size[1], self.spec.input_size[0]])]

    def get_providers(self):
        return list(self.providers)

    def run(self, output_names, feeds):
        assert output_names == [self.spec.output_name]
        assert list(feeds) == [self.spec.input_name]
        self.run_count += 1
        return [self.output]


class FakeRuntime:
    def __init__(self, spec, *, available=None, fail_accelerated=False, create_delay=0, session_options=None):
        self.spec = spec
        self.available = available or ["CPUExecutionProvider"]
        self.fail_accelerated = fail_accelerated
        self.create_delay = create_delay
        self.session_options = session_options or {}
        self.created_with = []
        self.sessions = []

    def get_available_providers(self):
        return list(self.available)

    def InferenceSession(self, path, providers):
        self.created_with.append((path, list(providers)))
        if self.create_delay:
            time.sleep(self.create_delay)
        if self.fail_accelerated and providers != ["CPUExecutionProvider"]:
            raise RuntimeError("accelerator unavailable")
        session = FakeSession(self.spec, providers, **self.session_options)
        self.sessions.append(session)
        return session


def installed_test_model(tmp_path, spec=None, filename="test.onnx"):
    spec = spec or adapter_spec(filename=filename)
    path = tmp_path / filename
    path.write_bytes(b"model")
    return background_removal.InstalledModel(spec=spec, path=path)


def test_runtime_is_loaded_lazily_and_accelerated_provider_precedes_cpu(tmp_path):
    model = installed_test_model(tmp_path)
    runtime = FakeRuntime(
        model.spec,
        available=["UnknownExecutionProvider", "CPUExecutionProvider", "CUDAExecutionProvider"],
    )
    load_count = 0

    def load_runtime():
        nonlocal load_count
        load_count += 1
        return runtime

    service = background_removal.BackgroundRemovalService(runtime_loader=load_runtime)
    assert load_count == 0

    result = service.remove_background(Image.new("RGB", (2, 2), "white"), model)

    assert result.mode == "RGBA"
    assert load_count == 1
    assert runtime.created_with == [(str(model.path.resolve()), ["CUDAExecutionProvider", "CPUExecutionProvider"])]


def test_runtime_retries_session_creation_once_with_cpu(tmp_path, caplog):
    model = installed_test_model(tmp_path)
    runtime = FakeRuntime(
        model.spec,
        available=["CUDAExecutionProvider", "CPUExecutionProvider"],
        fail_accelerated=True,
    )
    service = background_removal.BackgroundRemovalService(runtime_loader=lambda: runtime)

    with caplog.at_level(logging.WARNING, logger="modules.background_removal"):
        result = service.remove_background(Image.new("RGB", (2, 2)), model)

    assert result.mode == "RGBA"
    assert [providers for _path, providers in runtime.created_with] == [
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ["CPUExecutionProvider"],
    ]
    assert "retrying with CPU" in caplog.text


def test_runtime_reuses_session_until_model_identity_changes(tmp_path):
    first = installed_test_model(tmp_path, filename="first.onnx")
    runtime = FakeRuntime(first.spec)
    service = background_removal.BackgroundRemovalService(runtime_loader=lambda: runtime)
    image = Image.new("RGB", (2, 2))

    service.remove_background(image, first)
    service.remove_background(image, first)
    assert len(runtime.sessions) == 1
    assert runtime.sessions[0].run_count == 2

    second_spec = dataclasses.replace(first.spec, filename="second.onnx")
    second = installed_test_model(tmp_path, spec=second_spec, filename="second.onnx")
    service.remove_background(image, second)
    assert len(runtime.sessions) == 2

    second.path.write_bytes(b"replacement model with a different size")
    service.remove_background(image, second)
    assert len(runtime.sessions) == 3


def test_runtime_lock_prevents_duplicate_session_creation(tmp_path):
    model = installed_test_model(tmp_path)
    runtime = FakeRuntime(model.spec, create_delay=0.02)
    service = background_removal.BackgroundRemovalService(runtime_loader=lambda: runtime)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(
            lambda _index: service.remove_background(Image.new("RGB", (2, 2)), model),
            range(4),
        ))

    assert len(runtime.sessions) == 1
    assert all(result.mode == "RGBA" for result in results)


@pytest.mark.parametrize(
    "session_options",
    [
        {"input_name": "wrong_input"},
        {"input_shape": [1, 4, 2, 2]},
        {"output_name": "wrong_output"},
    ],
)
def test_runtime_contract_errors_identify_model_and_path(tmp_path, session_options):
    model = installed_test_model(tmp_path)
    runtime = FakeRuntime(model.spec, session_options=session_options)
    service = background_removal.BackgroundRemovalService(runtime_loader=lambda: runtime)

    with pytest.raises(background_removal.ModelContractError) as exc_info:
        service.remove_background(Image.new("RGB", (2, 2)), model)

    message = str(exc_info.value)
    assert model.spec.display_name in message
    assert str(model.path.resolve()) in message
