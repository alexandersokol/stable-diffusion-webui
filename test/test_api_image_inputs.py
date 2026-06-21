import base64
import io
import unittest

from fastapi.exceptions import HTTPException

from modules.api import image_inputs


class FakeResponse:
    def __init__(self, chunks=(), headers=None):
        self.headers = headers or {}
        self.chunks = list(chunks)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        yield from self.chunks


def data_url(raw):
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


class ApiImageInputTests(unittest.TestCase):
    def test_decode_base64_image_rejects_payload_before_decoding(self):
        read_calls = []

        with self.assertRaises(HTTPException) as caught:
            image_inputs.decode_base64_to_image(data_url(b"abcd"), max_bytes=3, read_image=lambda stream: read_calls.append(stream.read()))

        self.assertEqual(caught.exception.status_code, 413)
        self.assertEqual(read_calls, [])

    def test_decode_base64_image_accepts_small_payload(self):
        image = image_inputs.decode_base64_to_image(data_url(b"abcd"), max_bytes=4, read_image=lambda stream: stream.read())

        self.assertEqual(image, b"abcd")

    def test_download_image_rejects_content_length_over_limit_without_reading_body(self):
        def requests_get(*args, **kwargs):
            return FakeResponse(headers={"content-length": "4"}, chunks=[b"body"])

        with self.assertRaises(HTTPException) as caught:
            image_inputs.download_image_bytes("https://example.test/image.png", max_bytes=3, requests_get=requests_get)

        self.assertEqual(caught.exception.status_code, 413)

    def test_download_image_rejects_chunked_response_once_limit_exceeded(self):
        def requests_get(*args, **kwargs):
            return FakeResponse(chunks=[b"ab", b"cd"])

        with self.assertRaises(HTTPException) as caught:
            image_inputs.download_image_bytes("https://example.test/image.png", max_bytes=3, requests_get=requests_get)

        self.assertEqual(caught.exception.status_code, 413)

    def test_batch_budget_rejects_total_base64_input_before_decoding_next_image(self):
        budget = image_inputs.ApiImageInputBudget(max_bytes_per_image=10, max_batch_bytes=5)
        read_calls = []

        with self.assertRaises(HTTPException) as caught:
            image_inputs.decode_base64_batch_to_images(
                [data_url(b"abc"), data_url(b"def")],
                budget=budget,
                read_image=lambda stream: read_calls.append(stream.read()) or read_calls[-1],
            )

        self.assertEqual(caught.exception.status_code, 413)
        self.assertEqual(read_calls, [b"abc"])

    def test_batch_budget_allows_disabled_total_limit(self):
        budget = image_inputs.ApiImageInputBudget(max_bytes_per_image=10, max_batch_bytes=0)

        images = image_inputs.decode_base64_batch_to_images(
            [data_url(b"abc"), data_url(b"def")],
            budget=budget,
            read_image=lambda stream: stream.read(),
        )

        self.assertEqual(images, [b"abc", b"def"])


if __name__ == "__main__":
    unittest.main()
