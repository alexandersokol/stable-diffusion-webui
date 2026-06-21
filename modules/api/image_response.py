from fastapi.exceptions import HTTPException


API_IMAGE_RETURN_BASE64 = "base64"
API_IMAGE_RETURN_FILE = "file"
API_IMAGE_RETURN_MODES = {API_IMAGE_RETURN_BASE64, API_IMAGE_RETURN_FILE}


def normalize_api_image_return_mode(mode):
    mode = (mode or API_IMAGE_RETURN_BASE64).lower()
    if mode not in API_IMAGE_RETURN_MODES:
        raise HTTPException(status_code=422, detail=f"Invalid image_return_mode {mode!r}; expected one of {sorted(API_IMAGE_RETURN_MODES)}")

    return mode


def save_api_response_image(image, processed, image_index, sample_path, grid_path, save_image, samples_format, grid_format, grid_extended_filename):
    if isinstance(image, str):
        return image

    saved_path = getattr(image, "already_saved_as", None)
    if saved_path:
        return saved_path

    infotext = processed.infotexts[min(image_index, len(processed.infotexts) - 1)] if processed.infotexts else processed.info

    if image_index < processed.index_of_first_image:
        saved_path, _ = save_image(
            image,
            grid_path,
            "grid",
            processed.all_seeds[0] if processed.all_seeds else processed.seed,
            processed.all_prompts[0] if processed.all_prompts else processed.prompt,
            grid_format,
            info=infotext,
            short_filename=not grid_extended_filename,
            grid=True,
        )
    else:
        sample_index = image_index - processed.index_of_first_image
        seed = processed.all_seeds[min(sample_index, len(processed.all_seeds) - 1)] if processed.all_seeds else processed.seed
        prompt = processed.all_prompts[min(sample_index, len(processed.all_prompts) - 1)] if processed.all_prompts else processed.prompt
        saved_path, _ = save_image(image, sample_path, "", seed, prompt, samples_format, info=infotext)

    return saved_path


def encode_processed_images_for_api(
    processed,
    send_images,
    image_return_mode,
    sample_path,
    grid_path,
    encode_image,
    save_image,
    samples_format,
    grid_format,
    grid_extended_filename,
):
    if not send_images:
        return [], []

    image_return_mode = normalize_api_image_return_mode(image_return_mode)
    if image_return_mode == API_IMAGE_RETURN_BASE64:
        return list(map(encode_image, processed.images)), []

    image_paths = [
        save_api_response_image(image, processed, image_index, sample_path, grid_path, save_image, samples_format, grid_format, grid_extended_filename)
        for image_index, image in enumerate(processed.images)
    ]

    return [], image_paths
