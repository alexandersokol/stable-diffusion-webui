import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

from modules import profiling


def make_trace(path, size=10, mtime=1000):
    path.write_bytes(b"x" * size)
    os.utime(path, (mtime, mtime))
    return path


def test_rotate_existing_trace_keeps_configured_latest_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        trace = Path(tmpdir) / "trace.json"
        make_trace(trace, size=12)

        rotated = profiling.rotate_existing_trace(trace)

        assert rotated is not None
        assert rotated.exists()
        assert rotated.name.startswith("trace-")
        assert rotated.suffix == ".json"
        assert not trace.exists()


def test_cleanup_profile_traces_prunes_by_count():
    with tempfile.TemporaryDirectory() as tmpdir:
        trace = Path(tmpdir) / "trace.json"
        make_trace(trace, mtime=4000)
        oldest = make_trace(Path(tmpdir) / "trace-20200101-000000.json", mtime=1000)
        make_trace(Path(tmpdir) / "trace-20200102-000000.json", mtime=2000)
        make_trace(Path(tmpdir) / "trace-20200103-000000.json", mtime=3000)

        result = profiling.cleanup_profile_traces(trace, max_files=2, max_total_size_bytes=0)

        assert result["evicted"] == 2
        assert result["files"] == 2
        assert not oldest.exists()
        assert trace.exists()


def test_cleanup_profile_traces_prunes_by_total_size():
    with tempfile.TemporaryDirectory() as tmpdir:
        trace = Path(tmpdir) / "trace.json"
        make_trace(trace, size=8, mtime=3000)
        make_trace(Path(tmpdir) / "trace-20200101-000000.json", size=8, mtime=1000)
        make_trace(Path(tmpdir) / "trace-20200102-000000.json", size=8, mtime=2000)

        result = profiling.cleanup_profile_traces(trace, max_files=0, max_total_size_bytes=16)

        assert result["evicted"] == 1
        assert result["after"] <= 16
        assert trace.exists()


def test_cleanup_profile_traces_report_returns_reclaimed_size():
    original_opts = profiling.shared.opts
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            trace = Path(tmpdir) / "trace.json"
            make_trace(trace, size=10, mtime=2000)
            make_trace(Path(tmpdir) / "trace-20200101-000000.json", size=10, mtime=1000)
            profiling.shared.opts = SimpleNamespace(
                profiling_filename=str(trace),
                profiling_max_trace_files=1,
                profiling_max_total_size_mb=0,
            )

            report = profiling.cleanup_profile_traces_report()

            assert "removed 1 files" in report
            assert "reclaimed" in report
    finally:
        profiling.shared.opts = original_opts
