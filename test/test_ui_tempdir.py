import time
from pathlib import Path

from modules import shared, ui_tempdir


def test_cleanup_tmpdir_removes_only_tracked_files(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(shared.opts, "temp_dir", str(tmp_path), raising=False)

    owned_png = tmp_path / "owned.png"
    owned_txt = tmp_path / "owned.txt"
    unrelated_png = tmp_path / "user.png"

    owned_png.write_text("owned", encoding="utf8")
    owned_txt.write_text("owned", encoding="utf8")
    unrelated_png.write_text("user", encoding="utf8")

    ui_tempdir.register_tmp_file_ownership(str(owned_png))
    ui_tempdir.register_tmp_file_ownership(str(owned_txt))

    ui_tempdir.cleanup_tmpdr()

    assert not owned_png.exists()
    assert not owned_txt.exists()
    assert unrelated_png.exists()


def test_cleanup_tmpdir_removes_old_prefixed_orphans(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(shared.opts, "temp_dir", str(tmp_path), raising=False)

    old_orphan = tmp_path / f"{ui_tempdir.WEBUI_TEMP_FILE_PREFIX}old.png"
    recent_orphan = tmp_path / f"{ui_tempdir.WEBUI_TEMP_FILE_PREFIX}recent.png"
    old_user_png = tmp_path / "old-user.png"

    old_orphan.write_text("old", encoding="utf8")
    recent_orphan.write_text("recent", encoding="utf8")
    old_user_png.write_text("user", encoding="utf8")

    old_mtime = time.time() - ui_tempdir.WEBUI_TEMP_ORPHAN_TTL - 60
    old_orphan.touch()
    old_user_png.touch()
    monkeypatch.setattr(ui_tempdir.time, "time", lambda: old_mtime + ui_tempdir.WEBUI_TEMP_ORPHAN_TTL + 60)
    monkeypatch.setattr(ui_tempdir.os.path, "getmtime", lambda filename: old_mtime if filename in {str(old_orphan), str(old_user_png)} else time.time())

    ui_tempdir.cleanup_tmpdr()

    assert not old_orphan.exists()
    assert recent_orphan.exists()
    assert old_user_png.exists()
