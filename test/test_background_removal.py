import dataclasses
import logging

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
