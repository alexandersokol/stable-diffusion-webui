import base64
import binascii
import io
from dataclasses import dataclass

import requests
from fastapi.exceptions import HTTPException


STREAM_CHUNK_SIZE = 64 * 1024


def limit_bytes_from_mb(value):
    if value is None:
        return None

    value = float(value)
    if value <= 0:
        return None

    return int(value * 1024 * 1024)


def strip_base64_image_prefix(encoding):
    if encoding.startswith("data:image/"):
        try:
            return encoding.split(";", 1)[1].split(",", 1)[1]
        except Exception as e:
            raise HTTPException(status_code=500, detail="Invalid encoded image") from e

    return encoding


def estimate_base64_decoded_size(encoding):
    data = "".join(strip_base64_image_prefix(encoding).split())
    if not data:
        return 0

    padding = 0
    if data.endswith("=="):
        padding = 2
    elif data.endswith("="):
        padding = 1

    return max(0, len(data) * 3 // 4 - padding)


def check_input_size(size, max_bytes, detail):
    if max_bytes is not None and size > max_bytes:
        raise HTTPException(status_code=413, detail=detail)


def decode_base64_image_bytes(encoding, max_bytes=None):
    encoding = "".join(strip_base64_image_prefix(encoding).split())
    estimated_size = estimate_base64_decoded_size(encoding)
    check_input_size(estimated_size, max_bytes, "Encoded image is too large")

    try:
        decoded = base64.b64decode(encoding, validate=True)
    except (binascii.Error, ValueError) as e:
        raise HTTPException(status_code=500, detail="Invalid encoded image") from e

    check_input_size(len(decoded), max_bytes, "Encoded image is too large")
    return decoded


def download_image_bytes(url, max_bytes=None, requests_get=requests.get, headers=None, timeout=30, batch_budget=None):
    with requests_get(url, timeout=timeout, headers=headers or {}, stream=True) as response:
        response.raise_for_status()
        content_length = response.headers.get("content-length")
        if content_length:
            try:
                content_length = int(content_length)
                check_input_size(content_length, max_bytes, "Image URL is too large")
                if batch_budget is not None:
                    batch_budget.admit(content_length)
            except ValueError:
                pass

        data = io.BytesIO()
        total_size = 0
        for chunk in response.iter_content(chunk_size=STREAM_CHUNK_SIZE):
            if not chunk:
                continue

            total_size += len(chunk)
            check_input_size(total_size, max_bytes, "Image URL is too large")
            if batch_budget is not None and not content_length:
                batch_budget.admit_chunk(len(chunk))
            data.write(chunk)

        return data.getvalue()


def decode_base64_to_image(encoding, max_bytes=None, read_image=None):
    return read_image(io.BytesIO(decode_base64_image_bytes(encoding, max_bytes=max_bytes)))


def decode_url_to_image(url, max_bytes=None, read_image=None, requests_get=requests.get, headers=None):
    return read_image(io.BytesIO(download_image_bytes(url, max_bytes=max_bytes, requests_get=requests_get, headers=headers)))


@dataclass
class ApiImageInputBudget:
    max_bytes_per_image: int | None = None
    max_batch_bytes: int | None = None
    accepted_bytes: int = 0

    def __post_init__(self):
        if self.max_bytes_per_image is not None and self.max_bytes_per_image <= 0:
            self.max_bytes_per_image = None
        if self.max_batch_bytes is not None and self.max_batch_bytes <= 0:
            self.max_batch_bytes = None

    def admit(self, size):
        check_input_size(size, self.max_bytes_per_image, "Encoded image is too large")
        if self.max_batch_bytes is not None and self.accepted_bytes + size > self.max_batch_bytes:
            raise HTTPException(status_code=413, detail="API image batch is too large")

        self.accepted_bytes += size

    def admit_chunk(self, size):
        if self.max_batch_bytes is not None and self.accepted_bytes + size > self.max_batch_bytes:
            raise HTTPException(status_code=413, detail="API image batch is too large")

        self.accepted_bytes += size


def decode_base64_batch_to_images(encodings, budget, read_image):
    images = []
    for encoding in encodings:
        estimated_size = estimate_base64_decoded_size(encoding)
        budget.admit(estimated_size)
        images.append(decode_base64_to_image(encoding, max_bytes=budget.max_bytes_per_image, read_image=read_image))

    return images
