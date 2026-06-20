from PIL import Image


def _resize_image_for_restore(image, max_size):
    if not max_size or max_size <= 0:
        return image.copy()

    width, height = image.size
    largest_side = max(width, height)
    if largest_side <= max_size:
        return image.copy()

    scale = max_size / largest_side
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    resampling_filter = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    return image.resize(new_size, resampling_filter)


def _copy_lightweight_image(image, max_size):
    saved_path = getattr(image, "already_saved_as", None)
    if saved_path:
        placeholder = Image.new(mode=image.mode or "RGB", size=(1, 1))
        placeholder.already_saved_as = saved_path
        return placeholder

    restored = _resize_image_for_restore(image, max_size)
    restored.info = dict(getattr(image, "info", {}))
    return restored


def _copy_gallery_item(item, max_size):
    if isinstance(item, Image.Image):
        return _copy_lightweight_image(item, max_size)

    return item


def _copy_gallery(gallery, max_size):
    if not isinstance(gallery, (list, tuple)):
        return gallery

    copied = [_copy_gallery_item(item, max_size) for item in gallery]
    return tuple(copied) if isinstance(gallery, tuple) else copied


def create_lightweight_recorded_result(result, max_image_size=1024):
    if not isinstance(result, (list, tuple)) or not result:
        return result

    copied = list(result)
    copied[0] = _copy_gallery(copied[0], max_image_size)

    return tuple(copied) if isinstance(result, tuple) else copied
