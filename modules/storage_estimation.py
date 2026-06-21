from __future__ import annotations

import math
from dataclasses import dataclass


FORMAT_BYTES_PER_PIXEL = {
    "jpg": 1.0,
    "jpeg": 1.0,
    "webp": 1.2,
    "avif": 1.0,
    "png": 2.5,
}

TEXT_SIDECAR_BYTES = 4096


@dataclass
class GenerationStorageContext:
    width: int
    height: int
    batch_size: int = 1
    batch_count: int = 1
    tabname: str = "txt2img"
    enable_hr: bool = False
    init_image_count: int = 1
    has_mask: bool = False

    @property
    def image_count(self):
        return max(0, int(self.batch_size)) * max(0, int(self.batch_count))


@dataclass
class StorageSavingOptions:
    samples_save: bool = True
    samples_format: str = "png"
    grid_save: bool = True
    grid_format: str = "png"
    grid_only_if_multiple: bool = True
    save_txt: bool = False
    save_init_img: bool = False
    save_images_before_face_restoration: bool = False
    save_images_before_highres_fix: bool = False
    save_images_before_color_correction: bool = False
    save_mask: bool = False
    save_mask_composite: bool = False


@dataclass
class StorageEstimate:
    bytes: int
    image_files: int
    text_files: int


def estimate_image_file_bytes(width, height, image_format):
    width = max(1, int(width or 1))
    height = max(1, int(height or 1))
    bytes_per_pixel = FORMAT_BYTES_PER_PIXEL.get(str(image_format or "png").lower(), FORMAT_BYTES_PER_PIXEL["png"])

    return int(width * height * bytes_per_pixel)


def estimate_grid_dimensions(width, height, image_count, batch_size):
    if image_count <= 0:
        return 0, 0

    cols = max(1, min(int(batch_size or 1), image_count))
    rows = int(math.ceil(image_count / cols))

    return width * cols, height * rows


def estimate_generation_storage(context: GenerationStorageContext, options: StorageSavingOptions):
    total_bytes = 0
    image_files = 0
    text_files = 0
    image_count = context.image_count
    sample_bytes = estimate_image_file_bytes(context.width, context.height, options.samples_format)

    def add_images(count, bytes_per_image=sample_bytes):
        nonlocal total_bytes, image_files, text_files

        if count <= 0:
            return

        image_files += count
        total_bytes += int(count * bytes_per_image)
        if options.save_txt:
            text_files += count
            total_bytes += count * TEXT_SIDECAR_BYTES

    if options.samples_save:
        add_images(image_count)

        if options.save_images_before_face_restoration:
            add_images(image_count)

        if context.tabname == "txt2img" and context.enable_hr and options.save_images_before_highres_fix:
            add_images(image_count)

        if context.tabname == "img2img" and options.save_images_before_color_correction:
            add_images(image_count)

        if context.tabname == "img2img":
            if options.save_mask and context.has_mask:
                add_images(image_count)
            if options.save_mask_composite and context.has_mask:
                add_images(image_count)

    if context.tabname == "img2img" and options.save_init_img:
        add_images(max(1, context.init_image_count))

    grid_allowed = options.grid_save and image_count > 0 and (image_count > 1 or not options.grid_only_if_multiple)
    if grid_allowed:
        grid_width, grid_height = estimate_grid_dimensions(context.width, context.height, image_count, context.batch_size)
        add_images(1, estimate_image_file_bytes(grid_width, grid_height, options.grid_format))

    return StorageEstimate(bytes=total_bytes, image_files=image_files, text_files=text_files)


def format_bytes(num_bytes):
    if num_bytes >= 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024 * 1024):.1f} GB"

    return f"{num_bytes / (1024 * 1024):.1f} MB"
