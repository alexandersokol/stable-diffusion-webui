from pathlib import Path

import pytest

from modules.image_filename import get_next_available_filename


def test_get_next_available_filename_skips_existing_files(tmp_path: Path):
    (tmp_path / "00000-test.png").touch()
    (tmp_path / "00001-test.png").touch()

    filename = get_next_available_filename(str(tmp_path), "", "-test", "png", 0)

    assert filename == str(tmp_path / "00002-test.png")


def test_get_next_available_filename_fails_when_attempts_exhausted(tmp_path: Path):
    for index in range(3):
        (tmp_path / f"0000{index}-test.png").touch()

    with pytest.raises(FileExistsError):
        get_next_available_filename(str(tmp_path), "", "-test", "png", 0, max_attempts=3)
