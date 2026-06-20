from pathlib import Path

import pytest

from modules import model_merger_safety


def test_estimate_merge_memory_uses_peak_phase_sizes(tmp_path: Path):
    primary = tmp_path / "primary.safetensors"
    secondary = tmp_path / "secondary.safetensors"
    tertiary = tmp_path / "tertiary.safetensors"
    vae = tmp_path / "vae.pt"

    primary.write_bytes(b"a" * 100)
    secondary.write_bytes(b"b" * 50)
    tertiary.write_bytes(b"c" * 40)
    vae.write_bytes(b"d" * 20)

    required = model_merger_safety.estimate_merge_memory_required(
        str(primary),
        str(secondary),
        str(tertiary),
        str(vae),
    )

    assert required == int((100 + 50) * model_merger_safety.RAM_SAFETY_FACTOR)


def test_check_available_ram_raises_when_required_exceeds_available():
    with pytest.raises(MemoryError):
        model_merger_safety.check_available_ram(101, available_bytes=100)


def test_check_disk_space_raises_when_required_exceeds_available(tmp_path: Path):
    with pytest.raises(OSError):
        model_merger_safety.check_disk_space(str(tmp_path / "model.safetensors"), 100, free_bytes=104)


def test_atomic_write_replaces_output_on_success(tmp_path: Path):
    output = tmp_path / "model.safetensors"
    output.write_text("old", encoding="utf8")

    model_merger_safety.atomic_write(
        str(output),
        lambda temp_path: Path(temp_path).write_text("new", encoding="utf8"),
    )

    assert output.read_text(encoding="utf8") == "new"
    assert list(tmp_path.glob(".model.safetensors.*.tmp")) == []


def test_atomic_write_keeps_existing_output_and_removes_temp_on_failure(tmp_path: Path):
    output = tmp_path / "model.safetensors"
    output.write_text("old", encoding="utf8")

    def fail_after_writing(temp_path):
        Path(temp_path).write_text("partial", encoding="utf8")
        raise RuntimeError("save failed")

    with pytest.raises(RuntimeError):
        model_merger_safety.atomic_write(str(output), fail_after_writing)

    assert output.read_text(encoding="utf8") == "old"
    assert list(tmp_path.glob(".model.safetensors.*.tmp")) == []
