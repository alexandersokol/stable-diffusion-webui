import json
import os
import os.path
import threading
import time

import diskcache
import tqdm

from modules.paths import data_path, script_path

cache_filename = os.environ.get('SD_WEBUI_CACHE_FILE', os.path.join(data_path, "cache.json"))
cache_dir = os.environ.get('SD_WEBUI_CACHE_DIR', os.path.join(data_path, "cache"))
caches = {}
cache_lock = threading.RLock()
cache_cleanup_lock = threading.RLock()
cache_last_cleanup_time = 0

CACHE_SUBSECTION_SIZE_LIMIT = 2**32  # 4 GB, culling oldest first
DEFAULT_CACHE_MAX_SIZE_MB = 4096
DEFAULT_CACHE_CLEANUP_INTERVAL = 60
BYTES_IN_MB = 1024 * 1024


def dump_cache():
    """old function for dumping cache to disk; does nothing since diskcache."""

    pass


def make_cache(subsection: str) -> diskcache.Cache:
    return diskcache.Cache(
        os.path.join(cache_dir, subsection),
        size_limit=CACHE_SUBSECTION_SIZE_LIMIT,
        disk_min_file_size=2**18,  # keep up to 256KB in Sqlite
    )


def _get_option(name, default):
    try:
        from modules import shared
        return getattr(shared.opts, name, default)
    except Exception:
        return default


def format_bytes(num_bytes):
    if num_bytes < 1024:
        return f"{num_bytes} B"

    units = ["KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        value /= 1024.0
        if abs(value) < 1024.0:
            return f"{value:.1f} {unit}"

    return f"{value:.1f} PB"


def configured_cache_limit_bytes():
    max_size_mb = _get_option("disk_cache_max_size_mb", DEFAULT_CACHE_MAX_SIZE_MB)
    try:
        max_size_mb = int(max_size_mb)
    except (TypeError, ValueError):
        max_size_mb = DEFAULT_CACHE_MAX_SIZE_MB

    return max(0, max_size_mb) * BYTES_IN_MB


def configured_cleanup_interval():
    cleanup_interval = _get_option("disk_cache_cleanup_interval", DEFAULT_CACHE_CLEANUP_INTERVAL)
    try:
        cleanup_interval = int(cleanup_interval)
    except (TypeError, ValueError):
        cleanup_interval = DEFAULT_CACHE_CLEANUP_INTERVAL

    return max(0, cleanup_interval)


def list_cache_subsections():
    subsections = set(caches)

    if os.path.isdir(cache_dir):
        for name in os.listdir(cache_dir):
            full_path = os.path.join(cache_dir, name)
            if os.path.isdir(full_path):
                subsections.add(name)

    return sorted(subsections)


def _cache_for_cleanup(subsection):
    cache_obj = caches.get(subsection)
    if cache_obj is not None:
        return cache_obj, False

    return make_cache(subsection), True


def _close_temporary_cache(cache_obj, temporary):
    if temporary:
        cache_obj.close()


def get_cache_usage():
    usage = {}

    for subsection in list_cache_subsections():
        cache_obj, temporary = _cache_for_cleanup(subsection)
        try:
            cache_obj.expire()
            usage[subsection] = cache_obj.volume()
        finally:
            _close_temporary_cache(cache_obj, temporary)

    return usage


def _get_cache_entry_count(subsection):
    cache_obj, temporary = _cache_for_cleanup(subsection)
    try:
        return len(cache_obj)
    finally:
        _close_temporary_cache(cache_obj, temporary)


def _pop_oldest_cache_entry(subsection):
    cache_obj, temporary = _cache_for_cleanup(subsection)
    try:
        key, _ = cache_obj.peekitem(last=False, retry=True)
        cache_obj.pop(key, None, retry=True)
        return True
    except KeyError:
        return False
    finally:
        _close_temporary_cache(cache_obj, temporary)


def cleanup_cache(max_size_bytes=None, force=False, cleanup_interval=None):
    global cache_last_cleanup_time

    with cache_cleanup_lock:
        now = time.time()
        cleanup_interval = configured_cleanup_interval() if cleanup_interval is None else cleanup_interval

        if not force and cleanup_interval > 0 and now - cache_last_cleanup_time < cleanup_interval:
            return {
                "skipped": True,
                "limit": configured_cache_limit_bytes() if max_size_bytes is None else max_size_bytes,
                "before": None,
                "after": None,
                "evicted": 0,
                "usage": {},
            }

        cache_last_cleanup_time = now
        max_size_bytes = configured_cache_limit_bytes() if max_size_bytes is None else max_size_bytes
        usage_before = get_cache_usage()
        total_before = sum(usage_before.values())

        if max_size_bytes <= 0:
            return {
                "skipped": False,
                "limit": max_size_bytes,
                "before": total_before,
                "after": total_before,
                "evicted": 0,
                "usage": usage_before,
            }

        usage = dict(usage_before)
        total = total_before
        evicted = 0

        while total > max_size_bytes and usage:
            subsection = max(usage, key=usage.get)
            if _get_cache_entry_count(subsection) <= 0 or not _pop_oldest_cache_entry(subsection):
                usage.pop(subsection, None)
                continue

            evicted += 1
            usage = get_cache_usage()
            total = sum(usage.values())

        usage_after = get_cache_usage()
        return {
            "skipped": False,
            "limit": max_size_bytes,
            "before": total_before,
            "after": sum(usage_after.values()),
            "evicted": evicted,
            "usage": usage_after,
        }


def cleanup_cache_if_needed():
    try:
        return cleanup_cache(force=False)
    except Exception as e:
        print(f"Error cleaning disk cache: {e}")
        return {"skipped": True, "error": str(e)}


def cleanup_cache_report():
    try:
        result = cleanup_cache(force=True)
    except Exception as e:
        return f"Disk cache cleanup failed: {e}"

    if result["limit"] <= 0:
        return f"Disk cache cleanup is disabled. Current cache size: {format_bytes(result['after'])}."

    reclaimed = max(0, result["before"] - result["after"])
    return (
        f"Disk cache cleanup removed {result['evicted']} entries and reclaimed {format_bytes(reclaimed)}. "
        f"Cache size: {format_bytes(result['after'])} / {format_bytes(result['limit'])}."
    )


def convert_old_cached_data():
    try:
        with open(cache_filename, "r", encoding="utf8") as file:
            data = json.load(file)
    except FileNotFoundError:
        return
    except Exception:
        os.replace(cache_filename, os.path.join(script_path, "tmp", "cache.json"))
        print('[ERROR] issue occurred while trying to read cache.json; old cache has been moved to tmp/cache.json')
        return

    total_count = sum(len(keyvalues) for keyvalues in data.values())

    with tqdm.tqdm(total=total_count, desc="converting cache") as progress:
        for subsection, keyvalues in data.items():
            cache_obj = caches.get(subsection)
            if cache_obj is None:
                cache_obj = make_cache(subsection)
                caches[subsection] = cache_obj

            for key, value in keyvalues.items():
                cache_obj[key] = value
                progress.update(1)


def cache(subsection):
    """
    Retrieves or initializes a cache for a specific subsection.

    Parameters:
        subsection (str): The subsection identifier for the cache.

    Returns:
        diskcache.Cache: The cache data for the specified subsection.
    """

    cache_obj = caches.get(subsection)
    if not cache_obj:
        with cache_lock:
            if not os.path.exists(cache_dir) and os.path.isfile(cache_filename):
                convert_old_cached_data()

            cache_obj = caches.get(subsection)
            if not cache_obj:
                cache_obj = make_cache(subsection)
                caches[subsection] = cache_obj

    return cache_obj


def cached_data_for_file(subsection, title, filename, func):
    """
    Retrieves or generates data for a specific file, using a caching mechanism.

    Parameters:
        subsection (str): The subsection of the cache to use.
        title (str): The title of the data entry in the subsection of the cache.
        filename (str): The path to the file to be checked for modifications.
        func (callable): A function that generates the data if it is not available in the cache.

    Returns:
        dict or None: The cached or generated data, or None if data generation fails.

    The `cached_data_for_file` function implements a caching mechanism for data stored in files.
    It checks if the data associated with the given `title` is present in the cache and compares the
    modification time of the file with the cached modification time. If the file has been modified,
    the cache is considered invalid and the data is regenerated using the provided `func`.
    Otherwise, the cached data is returned.

    If the data generation fails, None is returned to indicate the failure. Otherwise, the generated
    or cached data is returned as a dictionary.
    """

    existing_cache = cache(subsection)
    ondisk_mtime = os.path.getmtime(filename)

    entry = existing_cache.get(title)
    if entry:
        cached_mtime = entry.get("mtime", 0)
        if ondisk_mtime > cached_mtime:
            entry = None

    if not entry or 'value' not in entry:
        value = func()
        if value is None:
            return None

        entry = {'mtime': ondisk_mtime, 'value': value}
        existing_cache[title] = entry

        dump_cache()
        cleanup_cache_if_needed()

    return entry['value']
