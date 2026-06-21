import os
import tempfile
import threading
import time
from pathlib import Path

from modules import cache


def use_temp_cache_dir():
    original_cache_dir = cache.cache_dir
    original_caches = cache.caches
    original_entry_locks = cache.cache_entry_locks
    original_last_cleanup = cache.cache_last_cleanup_time
    temp_dir = tempfile.TemporaryDirectory()

    cache.cache_dir = temp_dir.name
    cache.caches = {}
    cache.cache_entry_locks = {}
    cache.cache_last_cleanup_time = 0

    def restore():
        for cache_obj in cache.caches.values():
            cache_obj.close()
        cache.cache_dir = original_cache_dir
        cache.caches = original_caches
        cache.cache_entry_locks = original_entry_locks
        cache.cache_last_cleanup_time = original_last_cleanup
        temp_dir.cleanup()

    return Path(temp_dir.name), restore


def test_get_cache_usage_counts_all_subsection_directories():
    temp_cache_dir, restore = use_temp_cache_dir()
    try:
        metadata_cache = cache.cache("metadata")
        metadata_cache["large"] = "x" * 300000
        hash_cache = cache.cache("hashes")
        hash_cache["large"] = "y" * 300000

        usage = cache.get_cache_usage()

        assert set(usage) == {"hashes", "metadata"}
        assert usage["metadata"] > 0
        assert usage["hashes"] > 0
        assert temp_cache_dir.exists()
    finally:
        restore()


def test_cleanup_cache_evicts_entries_until_under_global_budget():
    _, restore = use_temp_cache_dir()
    try:
        metadata_cache = cache.cache("metadata")
        metadata_cache["first"] = "x" * 300000
        metadata_cache["second"] = "y" * 300000

        before = sum(cache.get_cache_usage().values())
        result = cache.cleanup_cache(max_size_bytes=max(1, before - 250000), force=True)

        assert result["evicted"] > 0
        assert result["after"] < result["before"]
        assert len(metadata_cache) < 2
    finally:
        restore()


def test_cleanup_cache_does_not_rescan_all_subsections_after_each_eviction():
    _, restore = use_temp_cache_dir()
    original_get_cache_usage = cache.get_cache_usage
    get_cache_usage_calls = 0

    def counting_get_cache_usage():
        nonlocal get_cache_usage_calls
        get_cache_usage_calls += 1
        return original_get_cache_usage()

    try:
        metadata_cache = cache.cache("metadata")
        for i in range(5):
            metadata_cache[f"entry-{i}"] = f"{i}" * 300000

        before = sum(original_get_cache_usage().values())
        cache.get_cache_usage = counting_get_cache_usage

        result = cache.cleanup_cache(max_size_bytes=max(1, before - 900000), force=True)

        assert result["evicted"] > 1
        assert result["after"] < result["before"]
        assert get_cache_usage_calls == 2
    finally:
        cache.get_cache_usage = original_get_cache_usage
        restore()


def test_cleanup_cache_disabled_budget_reports_without_eviction():
    _, restore = use_temp_cache_dir()
    try:
        metadata_cache = cache.cache("metadata")
        metadata_cache["first"] = "x" * 300000

        result = cache.cleanup_cache(max_size_bytes=0, force=True)

        assert result["evicted"] == 0
        assert result["after"] == result["before"]
        assert len(metadata_cache) == 1
    finally:
        restore()


def test_cleanup_cache_interval_skips_repeated_automatic_checks():
    _, restore = use_temp_cache_dir()
    try:
        metadata_cache = cache.cache("metadata")
        metadata_cache["first"] = "x" * 300000

        first = cache.cleanup_cache(max_size_bytes=1, force=False, cleanup_interval=60)
        second = cache.cleanup_cache(max_size_bytes=1, force=False, cleanup_interval=60)

        assert first["skipped"] is False
        assert second["skipped"] is True
    finally:
        restore()


def test_cached_data_for_file_runs_cleanup_after_new_write():
    temp_cache_dir, restore = use_temp_cache_dir()
    original_cleanup_cache_if_needed = cache.cleanup_cache_if_needed
    cleanup_calls = []

    try:
        source_file = temp_cache_dir / "source.txt"
        source_file.write_text("content", encoding="utf8")
        cache.cleanup_cache_if_needed = lambda: cleanup_calls.append("cleanup")

        value = cache.cached_data_for_file("metadata", "source", str(source_file), lambda: {"ok": True})
        cached_value = cache.cached_data_for_file("metadata", "source", str(source_file), lambda: {"ok": False})

        assert value == {"ok": True}
        assert cached_value == {"ok": True}
        assert cleanup_calls == ["cleanup"]
    finally:
        cache.cleanup_cache_if_needed = original_cleanup_cache_if_needed
        restore()


def test_cached_data_for_file_invalidates_when_size_changes_with_same_mtime():
    temp_cache_dir, restore = use_temp_cache_dir()
    original_cleanup_cache_if_needed = cache.cleanup_cache_if_needed

    try:
        source_file = temp_cache_dir / "source.txt"
        source_file.write_text("old", encoding="utf8")
        original_mtime = source_file.stat().st_mtime
        cache.cleanup_cache_if_needed = lambda: None

        first = cache.cached_data_for_file("metadata", "source", str(source_file), lambda: {"value": "old"})

        source_file.write_text("new content", encoding="utf8")
        os.utime(source_file, (original_mtime, original_mtime))
        second = cache.cached_data_for_file("metadata", "source", str(source_file), lambda: {"value": "new"})

        assert first == {"value": "old"}
        assert second == {"value": "new"}
    finally:
        cache.cleanup_cache_if_needed = original_cleanup_cache_if_needed
        restore()


def test_cached_data_for_file_accepts_legacy_mtime_only_entry():
    temp_cache_dir, restore = use_temp_cache_dir()

    try:
        source_file = temp_cache_dir / "source.txt"
        source_file.write_text("content", encoding="utf8")
        metadata_cache = cache.cache("metadata")
        metadata_cache["source"] = {"mtime": source_file.stat().st_mtime + 1, "value": {"ok": True}}

        value = cache.cached_data_for_file("metadata", "source", str(source_file), lambda: {"ok": False})

        assert value == {"ok": True}
    finally:
        restore()


def test_cached_data_for_file_deduplicates_concurrent_misses():
    temp_cache_dir, restore = use_temp_cache_dir()
    original_cleanup_cache_if_needed = cache.cleanup_cache_if_needed
    call_count = 0
    call_lock = threading.Lock()
    start_barrier = threading.Barrier(4)
    results = []

    def read_metadata():
        nonlocal call_count
        with call_lock:
            call_count += 1
        time.sleep(0.05)
        return {"value": "computed"}

    def worker(source_file):
        start_barrier.wait()
        results.append(cache.cached_data_for_file("metadata", "source", str(source_file), read_metadata))

    try:
        source_file = temp_cache_dir / "source.txt"
        source_file.write_text("content", encoding="utf8")
        cache.cleanup_cache_if_needed = lambda: None
        threads = [threading.Thread(target=worker, args=(source_file,)) for _ in range(4)]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert call_count == 1
        assert results == [{"value": "computed"}] * 4
    finally:
        cache.cleanup_cache_if_needed = original_cleanup_cache_if_needed
        restore()
