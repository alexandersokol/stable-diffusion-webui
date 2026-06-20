from PIL import Image


def create_output_image_placeholder(saved_path):
    image = Image.new(mode="RGB", size=(1, 1))
    image.already_saved_as = saved_path
    return image


def replace_saved_images_with_placeholders(p, processed, saved_image_paths):
    if p.is_api or len(processed.images) != len(saved_image_paths):
        return

    for index, saved_path in enumerate(saved_image_paths):
        if saved_path:
            processed.images[index] = create_output_image_placeholder(saved_path)
