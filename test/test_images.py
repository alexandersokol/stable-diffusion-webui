from pathlib import Path

from modules.image_filename import get_next_available_filename, get_next_sequence_filename, reset_sequence_number_cache


def test_get_next_available_filename_skips_existing_files(tmp_path: Path):
    (tmp_path / "00000-test.png").touch()
    (tmp_path / "00001-test.png").touch()

    filename = get_next_available_filename(str(tmp_path), "", "-test", "png", 0)

    assert filename == str(tmp_path / "00002-test.png")


def test_get_next_available_filename_fails_when_attempts_exhausted(tmp_path: Path):
    for index in range(3):
        (tmp_path / f"0000{index}-test.png").touch()

    try:
        get_next_available_filename(str(tmp_path), "", "-test", "png", 0, max_attempts=3)
    except FileExistsError:
        pass
    else:
        raise AssertionError("expected filename allocation to fail after exhausting attempts")


def test_get_next_sequence_filename_scans_once_then_uses_cache(tmp_path: Path):
    reset_sequence_number_cache()
    (tmp_path / "00000-test.png").touch()
    (tmp_path / "00001-test.png").touch()

    first = get_next_sequence_filename(str(tmp_path), "", "-test", "png")
    second = get_next_sequence_filename(str(tmp_path), "", "-test", "png")

    assert first == str(tmp_path / "00002-test.png")
    assert second == str(tmp_path / "00003-test.png")


def test_get_next_sequence_filename_skips_external_collision(tmp_path: Path):
    reset_sequence_number_cache()
    (tmp_path / "00000-test.png").touch()

    first = get_next_sequence_filename(str(tmp_path), "", "-test", "png")
    (tmp_path / "00002-test.png").touch()
    second = get_next_sequence_filename(str(tmp_path), "", "-test", "png")

    assert first == str(tmp_path / "00001-test.png")
    assert second == str(tmp_path / "00003-test.png")


def test_get_next_sequence_filename_rescans_when_cache_is_stale(tmp_path: Path):
    reset_sequence_number_cache()

    first = get_next_sequence_filename(str(tmp_path), "", "-test", "png")
    for index in range(1, 4):
        (tmp_path / f"0000{index}-test.png").touch()

    second = get_next_sequence_filename(str(tmp_path), "", "-test", "png", max_attempts=2)

    assert first == str(tmp_path / "00000-test.png")
    assert second == str(tmp_path / "00004-test.png")


def test_get_next_sequence_filename_uses_separate_cache_per_basename(tmp_path: Path):
    reset_sequence_number_cache()
    (tmp_path / "grid-0000-test.png").touch()
    (tmp_path / "00000-test.png").touch()

    grid = get_next_sequence_filename(str(tmp_path), "grid", "-test", "png")
    image = get_next_sequence_filename(str(tmp_path), "", "-test", "png")

    assert grid == str(tmp_path / "grid-0001-test.png")
    assert image == str(tmp_path / "00001-test.png")
