import os
import tempfile
from types import SimpleNamespace

from modules import output_storage


def write_file(path, size_or_text):
    if isinstance(size_or_text, int):
        with open(path, "wb") as file:
            file.write(b"x" * size_or_text)
    else:
        with open(path, "w", encoding="utf8") as file:
            file.write(size_or_text)


def read_file(path):
    with open(path, "r", encoding="utf8") as file:
        return file.read()


def test_rotate_log_file_shifts_backups_and_prunes_oldest():
    with tempfile.TemporaryDirectory() as temp_dir:
        logfile = os.path.join(temp_dir, "log.csv")
        write_file(logfile, "current")
        write_file(f"{logfile}.1", "backup1")
        write_file(f"{logfile}.2", "backup2")

        result = output_storage.rotate_log_file_if_needed(logfile, max_size_bytes=1, backup_count=2)

        assert result["rotated"] is True
        assert not os.path.exists(logfile)
        assert read_file(f"{logfile}.1") == "current"
        assert read_file(f"{logfile}.2") == "backup1"


def test_rotate_log_file_skips_when_disabled():
    with tempfile.TemporaryDirectory() as temp_dir:
        logfile = os.path.join(temp_dir, "log.csv")
        write_file(logfile, 10)

        result = output_storage.rotate_log_file_if_needed(logfile, max_size_bytes=0, backup_count=2)

        assert result["rotated"] is False
        assert os.path.exists(logfile)


def test_rotate_log_file_with_zero_backups_removes_current_log():
    with tempfile.TemporaryDirectory() as temp_dir:
        logfile = os.path.join(temp_dir, "log.csv")
        write_file(logfile, 10)

        result = output_storage.rotate_log_file_if_needed(logfile, max_size_bytes=1, backup_count=0)

        assert result["rotated"] is True
        assert not os.path.exists(logfile)


def test_summarize_output_storage_counts_images_txt_logs_and_folders():
    with tempfile.TemporaryDirectory() as temp_dir:
        first = os.path.join(temp_dir, "first")
        second = os.path.join(temp_dir, "second")
        os.makedirs(first)
        os.makedirs(second)
        write_file(os.path.join(first, "image.png"), 10)
        write_file(os.path.join(first, "image.txt"), 3)
        write_file(os.path.join(second, "log.csv"), 5)
        write_file(os.path.join(second, "note.bin"), 7)

        summary = output_storage.summarize_output_storage([first, second, os.path.join(temp_dir, "missing")])

        assert summary["total_files"] == 4
        assert summary["images_files"] == 1
        assert summary["images_bytes"] == 10
        assert summary["txt_files"] == 1
        assert summary["txt_bytes"] == 3
        assert summary["logs_files"] == 1
        assert summary["logs_bytes"] == 5
        assert summary["other_files"] == 1
        assert summary["missing_dirs"] == [os.path.join(temp_dir, "missing")]
        assert summary["top_dirs"][0][0] == first


def test_output_storage_report_escapes_paths_and_uses_configured_dirs():
    original_configured_output_dirs = output_storage.configured_output_dirs
    original_summarize_output_storage = output_storage.summarize_output_storage

    try:
        output_storage.configured_output_dirs = lambda: ["ignored"]
        output_storage.summarize_output_storage = lambda paths: {
            "total_files": 2,
            "total_bytes": 12,
            "images_files": 1,
            "images_bytes": 10,
            "txt_files": 1,
            "txt_bytes": 2,
            "logs_files": 0,
            "logs_bytes": 0,
            "missing_dirs": ["D:\\bad<name>"],
            "top_dirs": [("D:\\out<unsafe>", 12)],
        }

        report = output_storage.output_storage_report()

        assert "Output storage: 12 B in 2 files." in report
        assert "D:\\out&lt;unsafe&gt;" in report
        assert "D:\\bad&lt;name&gt;" in report
    finally:
        output_storage.configured_output_dirs = original_configured_output_dirs
        output_storage.summarize_output_storage = original_summarize_output_storage


def test_configured_log_options_are_normalized():
    opts = SimpleNamespace(save_log_max_size_mb="2.5", save_log_backup_count="3")

    assert output_storage.configured_log_max_size_bytes(opts) == int(2.5 * 1024 * 1024)
    assert output_storage.configured_log_backup_count(opts) == 3
