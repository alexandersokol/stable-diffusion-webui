import tempfile
from pathlib import Path

from modules import cache


def use_temp_cache_dir():
    original_cache_dir = cache.cache_dir
    original_caches = cache.caches
    original_last_cleanup = cache.cache_last_cleanup_time
    temp_dir = tempfile.TemporaryDirectory()

    cache.cache_dir = temp_dir.name
    cache.caches = {}
    cache.cache_last_cleanup_time = 0

    def restore():
        for cache_obj in cache.caches.values():
            cache_obj.close()
        cache.cache_dir = original_cache_dir
        cache.caches = original_caches
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
