import tempfile
from pathlib import Path

from modules import maintenance


def test_cleanup_removes_gradio_theme_json_only():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        theme_dir = root / "tmp" / "gradio_themes"
        theme_dir.mkdir(parents=True)
        theme_json = theme_dir / "theme.json"
        unrelated = theme_dir / "notes.txt"
        theme_json.write_text("{}", encoding="utf8")
        unrelated.write_text("keep", encoding="utf8")

        result = maintenance.cleanup_auxiliary_temp_files(root, include_ui_tempdir=False)

        assert result["removed"] == 1
        assert not theme_json.exists()
        assert unrelated.exists()


def test_cleanup_removes_moved_config_and_cache_only():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        tmp_path = root / "tmp"
        tmp_path.mkdir()
        moved_config = tmp_path / "config.json"
        moved_cache = tmp_path / "cache.json"
        unrelated = tmp_path / "other.json"
        moved_config.write_text("bad config", encoding="utf8")
        moved_cache.write_text("bad cache", encoding="utf8")
        unrelated.write_text("keep", encoding="utf8")

        result = maintenance.cleanup_auxiliary_temp_files(root, include_ui_tempdir=False)

        assert result["removed"] == 2
        assert not moved_config.exists()
        assert not moved_cache.exists()
        assert unrelated.exists()


def test_cleanup_report_includes_tempdir_result():
    original_cleanup = maintenance.ui_tempdir.cleanup_tmpdir_report
    original_cleanup_auxiliary = maintenance.cleanup_auxiliary_temp_files
    try:
        maintenance.ui_tempdir.cleanup_tmpdir_report = lambda: "WebUI temp cleanup removed 1 files and reclaimed 1.0 MB."
        maintenance.cleanup_auxiliary_temp_files = lambda: {
            "removed": 2,
            "reclaimed": 1024 * 1024,
            "ui_tempdir": maintenance.ui_tempdir.cleanup_tmpdir_report(),
        }

        report = maintenance.cleanup_auxiliary_temp_files_report()

        assert "removed 2 files" in report
        assert "1.0 MB" in report
        assert "WebUI temp cleanup removed 1 files" in report
    finally:
        maintenance.ui_tempdir.cleanup_tmpdir_report = original_cleanup
        maintenance.cleanup_auxiliary_temp_files = original_cleanup_auxiliary
