from types import SimpleNamespace

import pytest
from PIL import Image

from modules import shared
from modules import background_removal, scripts_postprocessing
from scripts import postprocessing_background_removal


def configure_model_path(monkeypatch, path):
    monkeypatch.setattr(
        postprocessing_background_removal.shared,
        "cmd_opts",
        SimpleNamespace(background_removal_models_path=str(path)),
    )


def test_script_is_a_terminal_background_removal_operation():
    script = postprocessing_background_removal.ScriptPostprocessingBackgroundRemoval()

    assert script.name == "Background Removal"
    assert script.order > 4000
    assert script.terminal is True


def test_model_choices_include_only_files_below_configured_path(monkeypatch, tmp_path):
    configure_model_path(monkeypatch, tmp_path)
    (tmp_path / "RMBG-1.4.onnx").write_bytes(b"installed")
    (tmp_path / "unknown.onnx").write_bytes(b"ignored")
    script = postprocessing_background_removal.ScriptPostprocessingBackgroundRemoval()

    assert script.model_choices() == ["RMBG 1.4"]


def test_disabled_processing_is_a_noop(monkeypatch, tmp_path):
    configure_model_path(monkeypatch, tmp_path)
    original = Image.new("RGB", (2, 2), "red")
    pp = scripts_postprocessing.PostprocessedImage(original)
    script = postprocessing_background_removal.ScriptPostprocessingBackgroundRemoval()

    script.process(pp, enabled=False, model_name=None)

    assert pp.image is original
    assert pp.info == {}
    assert pp.output_extension is None
    assert script.output_extension("jpg", enabled=False, model_name=None) == "jpg"


@pytest.mark.parametrize("model_name", [None, "", "Not Installed"])
def test_enabled_processing_rejects_an_unavailable_selection(monkeypatch, tmp_path, model_name):
    configure_model_path(monkeypatch, tmp_path)
    pp = scripts_postprocessing.PostprocessedImage(Image.new("RGB", (2, 2)))
    script = postprocessing_background_removal.ScriptPostprocessingBackgroundRemoval()

    with pytest.raises(background_removal.BackgroundRemovalError, match="model"):
        script.process(pp, enabled=True, model_name=model_name)


def test_enabled_processing_rejects_a_model_removed_after_selection(monkeypatch, tmp_path):
    configure_model_path(monkeypatch, tmp_path)
    model_path = tmp_path / "RMBG-1.4.onnx"
    model_path.write_bytes(b"installed")
    script = postprocessing_background_removal.ScriptPostprocessingBackgroundRemoval()
    assert script.model_choices() == ["RMBG 1.4"]
    model_path.unlink()
    pp = scripts_postprocessing.PostprocessedImage(Image.new("RGB", (2, 2)))

    with pytest.raises(background_removal.BackgroundRemovalError, match="RMBG 1.4"):
        script.process(pp, enabled=True, model_name="RMBG 1.4")


def test_enabled_processing_sets_rgba_info_and_png_output(monkeypatch, tmp_path):
    configure_model_path(monkeypatch, tmp_path)
    (tmp_path / "RMBG-1.4.onnx").write_bytes(b"installed")
    result = Image.new("RGBA", (2, 2), (10, 20, 30, 40))

    class FakeService:
        def remove_background(self, image, model):
            assert image.mode == "RGB"
            assert model.spec.display_name == "RMBG 1.4"
            return result

    monkeypatch.setattr(postprocessing_background_removal, "service", FakeService())
    pp = scripts_postprocessing.PostprocessedImage(Image.new("RGB", (2, 2)))
    script = postprocessing_background_removal.ScriptPostprocessingBackgroundRemoval()

    script.process(pp, enabled=True, model_name="RMBG 1.4")

    assert pp.image is result
    assert pp.info["Background removal model"] == "RMBG 1.4"
    assert pp.output_extension == "png"
    assert script.output_extension("webp", enabled=True, model_name="RMBG 1.4") == "png"
