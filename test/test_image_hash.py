import hashlib
import unittest

from PIL import Image

from modules.image_hash import calculate_image_sha256


class FilenameGeneratorImageHashTest(unittest.TestCase):
    def test_image_hash_avoids_full_image_tobytes(self):
        image = Image.new("RGB", (3, 2))
        image.putdata([
            (0, 1, 2), (3, 4, 5), (6, 7, 8),
            (9, 10, 11), (12, 13, 14), (15, 16, 17),
        ])
        expected = hashlib.sha256(image.tobytes()).hexdigest()

        def fail_full_image_tobytes(*_args, **_kwargs):
            raise AssertionError("full image tobytes should not be used")

        image.tobytes = fail_full_image_tobytes

        actual = calculate_image_sha256(image)

        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
