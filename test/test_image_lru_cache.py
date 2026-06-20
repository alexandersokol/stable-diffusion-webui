import os
from pathlib import Path

from PIL import Image

from modules import image_lru_cache


def test_image_lru_cache_evicts_by_item_count():
    cache = image_lru_cache.ImageLRUCache()
    first = Image.new("RGB", (8, 8), "red")
    second = Image.new("RGB", (8, 8), "blue")

    cache.put("first", first, max_items=1, max_bytes=0)
    cache.put("second", second, max_items=1, max_bytes=0)

    assert cache.get("first") is None
    assert cache.get("second") is second


def test_image_lru_cache_evicts_by_byte_budget():
    cache = image_lru_cache.ImageLRUCache()
    first = Image.new("RGB", (16, 16), "red")
    second = Image.new("RGB", (16, 16), "blue")
    max_bytes = image_lru_cache.estimate_image_bytes(second) + 1

    cache.put("first", first, max_items=10, max_bytes=max_bytes)
    cache.put("second", second, max_items=10, max_bytes=max_bytes)

    assert cache.get("first") is None
    assert cache.get("second") is second
    assert cache.total_bytes <= max_bytes


def test_image_lru_cache_zero_byte_budget_keeps_count_limited_entries():
    cache = image_lru_cache.ImageLRUCache()
    first = Image.new("RGB", (16, 16), "red")
    second = Image.new("RGB", (16, 16), "blue")

    cache.put("first", first, max_items=2, max_bytes=0)
    cache.put("second", second, max_items=2, max_bytes=0)

    assert cache.get("first") is first
    assert cache.get("second") is second


def test_image_lru_cache_zero_items_disables_cache():
    cache = image_lru_cache.ImageLRUCache()

    cache.put("first", Image.new("RGB", (8, 8), "red"), max_items=0, max_bytes=0)

    assert cache.get("first") is None
    assert cache.total_bytes == 0


def test_image_cache_source_key_uses_file_metadata(tmp_path: Path):
    filename = tmp_path / "source.png"
    Image.new("RGB", (8, 8), "red").save(filename)

    with Image.open(filename) as image:
        key = image_lru_cache.image_cache_source_key(image)

    assert key[0] == "file"
    assert key[1] == os.path.abspath(filename)
    assert key[3] == os.path.getsize(filename)
    assert key[4] == "RGB"
    assert key[5] == (8, 8)
