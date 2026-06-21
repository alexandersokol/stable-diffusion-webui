import unittest
import sys
import types
from unittest import mock

from PIL import Image, ImageChops

sys.modules.setdefault("modules.sd_samplers", types.ModuleType("modules.sd_samplers"))

from modules import images


def make_gradient_image(width=9, height=8):
    image = Image.new("RGB", (width, height))
    pixels = image.load()

    for y in range(height):
        for x in range(width):
            pixels[x, y] = ((x * 23) % 256, (y * 29) % 256, ((x + y) * 17) % 256)

    return image


def legacy_combine_grid(grid):
    import numpy as np

    def make_mask_image(r):
        r = r * 255 / grid.overlap
        r = r.astype(np.uint8)
        return Image.fromarray(r, 'L')

    mask_w = make_mask_image(np.arange(grid.overlap, dtype=np.float32).reshape((1, grid.overlap)).repeat(grid.tile_h, axis=0))
    mask_h = make_mask_image(np.arange(grid.overlap, dtype=np.float32).reshape((grid.overlap, 1)).repeat(grid.image_w, axis=1))

    combined_image = Image.new("RGB", (grid.image_w, grid.image_h))
    for y, h, row in grid.tiles:
        combined_row = Image.new("RGB", (grid.image_w, h))
        for x, w, tile in row:
            if x == 0:
                combined_row.paste(tile, (0, 0))
                continue

            combined_row.paste(tile.crop((0, 0, grid.overlap, h)), (x, 0), mask=mask_w)
            combined_row.paste(tile.crop((grid.overlap, 0, w, h)), (x + grid.overlap, 0))

        if y == 0:
            combined_image.paste(combined_row, (0, 0))
            continue

        combined_image.paste(combined_row.crop((0, 0, combined_row.width, grid.overlap)), (0, y), mask=mask_h)
        combined_image.paste(combined_row.crop((0, grid.overlap, combined_row.width, h)), (0, y + grid.overlap))

    return combined_image


class ImageGridMemoryTest(unittest.TestCase):
    def test_split_grid_defers_tile_crop_until_tile_is_read(self):
        source = make_gradient_image()

        with mock.patch.object(source, "crop", wraps=source.crop) as crop:
            grid = images.split_grid(source, tile_w=4, tile_h=4, overlap=2)

            self.assertEqual(0, crop.call_count)
            first_tile = grid.tiles[0][2][0]

            tile_image = first_tile[2]
            self.assertEqual((4, 4), tile_image.size)
            self.assertEqual(1, crop.call_count)

            self.assertIs(tile_image, first_tile[2])
            self.assertEqual(1, crop.call_count)

            replacement = Image.new("RGB", (4, 4), "red")
            first_tile[2] = replacement
            self.assertIs(replacement, first_tile[2])
            self.assertEqual(1, crop.call_count)

    def test_combine_grid_matches_legacy_output_after_tile_updates(self):
        source = make_gradient_image()
        grid = images.split_grid(source, tile_w=4, tile_h=4, overlap=2)

        for row_index, (_y, _h, row) in enumerate(grid.tiles):
            for col_index, tiledata in enumerate(row):
                tile = tiledata[2].copy()
                if (row_index + col_index) % 2:
                    tile = ImageChops.invert(tile)
                tiledata[2] = tile

        expected = legacy_combine_grid(grid)
        actual = images.combine_grid(grid)

        self.assertIsNone(ImageChops.difference(expected, actual).getbbox())


if __name__ == "__main__":
    unittest.main()
