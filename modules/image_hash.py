import hashlib


IMAGE_HASH_CHUNK_BYTES = 1024 * 1024


def calculate_image_sha256(image):
    hash_sha256 = hashlib.sha256()
    width, height = image.size

    if width == 0 or height == 0:
        return hash_sha256.hexdigest()

    first_row = image.crop((0, 0, width, 1)).tobytes()
    bytes_per_row = max(len(first_row), 1)
    rows_per_chunk = max(1, IMAGE_HASH_CHUNK_BYTES // bytes_per_row)

    if rows_per_chunk == 1:
        hash_sha256.update(first_row)
        start_row = 1
    else:
        start_row = 0

    for y in range(start_row, height, rows_per_chunk):
        hash_sha256.update(image.crop((0, y, width, min(height, y + rows_per_chunk))).tobytes())

    return hash_sha256.hexdigest()
