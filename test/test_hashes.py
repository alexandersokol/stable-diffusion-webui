import builtins
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from modules import cache, hashes


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


class HashesTests(unittest.TestCase):
    def test_legacy_model_hash_uses_file_signature_cache(self):
        temp_cache_dir, restore = use_temp_cache_dir()
        original_cleanup_cache_if_needed = cache.cleanup_cache_if_needed

        try:
            payload = b"legacy-hash-payload"
            model_file = temp_cache_dir / "model.ckpt"
            model_file.write_bytes((b"\0" * 0x100000) + payload)
            expected = hashlib.sha256(payload).hexdigest()[0:8]
            cache.cleanup_cache_if_needed = lambda: None

            first = hashes.legacy_model_hash(str(model_file), "checkpoint/model.ckpt")

            def open_without_model_file(filename, *args, **kwargs):
                if filename == str(model_file):
                    raise AssertionError("legacy model hash should be loaded from cache")
                return builtins.open(filename, *args, **kwargs)

            with mock.patch("builtins.open", open_without_model_file):
                second = hashes.legacy_model_hash(str(model_file), "checkpoint/model.ckpt")

            self.assertEqual(first, expected)
            self.assertEqual(second, expected)
        finally:
            cache.cleanup_cache_if_needed = original_cleanup_cache_if_needed
            restore()


def test_legacy_model_hash_uses_file_signature_cache():
    HashesTests("test_legacy_model_hash_uses_file_signature_cache").test_legacy_model_hash_uses_file_signature_cache()


if __name__ == "__main__":
    unittest.main()
