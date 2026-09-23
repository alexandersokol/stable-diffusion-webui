from PIL import Image


def create_output_image_placeholder(saved_path):
    image = Image.new(mode="RGB", size=(1, 1))
    image.already_saved_as = saved_path
    return image


def should_store_saved_image_as_placeholder(p, opts, saved_path):
    if not saved_path:
        return False

    if getattr(p, "is_api", False):
        return False

    if not getattr(opts, "drop_saved_images_from_results", False):
        return False

    if not getattr(opts, "do_not_show_images", False):
        return False

    return not getattr(opts, "return_grid", False) and not getattr(opts, "grid_save", False)


def append_image_result(output_images, output_image_paths, p, opts, image, saved_path):
    if should_store_saved_image_as_placeholder(p, opts, saved_path):
        output_images.append(create_output_image_placeholder(saved_path))
        output_image_paths.append(saved_path)
        image.close()
        return

    output_images.append(image)
    output_image_paths.append(saved_path)


def replace_saved_images_with_placeholders(p, opts, processed, saved_image_paths):
    if p.is_api or len(processed.images) != len(saved_image_paths):
        return

    for index, saved_path in enumerate(saved_image_paths):
        if should_store_saved_image_as_placeholder(p, opts, saved_path):
            processed.images[index] = create_output_image_placeholder(saved_path)
