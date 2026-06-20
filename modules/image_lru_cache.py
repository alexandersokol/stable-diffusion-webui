import os
from collections import OrderedDict


def estimate_image_bytes(image):
    bands = len(image.getbands()) if hasattr(image, "getbands") else len(getattr(image, "mode", "RGB"))
    return int(image.width * image.height * max(1, bands))


def image_cache_source_key(image):
    filename = getattr(image, "filename", None)
    if filename and os.path.isfile(filename):
        stat = os.stat(filename)
        return ("file", os.path.abspath(filename), stat.st_mtime_ns, stat.st_size, image.mode, image.size)

    sample = []
    if image.width and image.height:
        points = {
            (0, 0),
            (image.width // 2, image.height // 2),
            (image.width - 1, image.height - 1),
        }
        for point in sorted(points):
            try:
                sample.append((point, image.getpixel(point)))
            except Exception:
                pass

    return ("memory", id(image), image.mode, image.size, tuple(sample))


class ImageLRUCache:
    def __init__(self):
        self.items = OrderedDict()
        self.total_bytes = 0

    def clear(self):
        self.items.clear()
        self.total_bytes = 0

    def get(self, key):
        cached = self.items.pop(key, None)
        if cached is None:
            return None

        image, size_bytes = cached
        self.items[key] = (image, size_bytes)
        return image

    def put(self, key, image, max_items, max_bytes):
        max_items = max(0, int(max_items or 0))
        max_bytes = max(0, int(max_bytes or 0))
        if max_items == 0:
            self.clear()
            return

        old = self.items.pop(key, None)
        if old is not None:
            self.total_bytes -= old[1]

        size_bytes = estimate_image_bytes(image)
        self.items[key] = (image, size_bytes)
        self.total_bytes += size_bytes

        self._evict(max_items, max_bytes)

    def _evict(self, max_items, max_bytes):
        while len(self.items) > max_items:
            _, (_, size_bytes) = self.items.popitem(last=False)
            self.total_bytes -= size_bytes

        if max_bytes <= 0:
            return

        while len(self.items) > 1 and self.total_bytes > max_bytes:
            _, (_, size_bytes) = self.items.popitem(last=False)
            self.total_bytes -= size_bytes
