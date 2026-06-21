from modules import storage_estimation


def test_estimates_saved_samples_and_txt_sidecars():
    context = storage_estimation.GenerationStorageContext(width=512, height=512, batch_size=2, batch_count=3)
    options = storage_estimation.StorageSavingOptions(samples_save=True, grid_save=False, save_txt=True)

    estimate = storage_estimation.estimate_generation_storage(context, options)

    assert estimate.image_files == 6
    assert estimate.text_files == 6
    assert estimate.bytes == 6 * storage_estimation.estimate_image_file_bytes(512, 512, "png") + 6 * storage_estimation.TEXT_SIDECAR_BYTES


def test_estimates_grid_as_one_large_image():
    context = storage_estimation.GenerationStorageContext(width=512, height=768, batch_size=2, batch_count=3)
    options = storage_estimation.StorageSavingOptions(samples_save=False, grid_save=True, save_txt=False)

    estimate = storage_estimation.estimate_generation_storage(context, options)
    grid_width, grid_height = storage_estimation.estimate_grid_dimensions(512, 768, 6, 2)

    assert (grid_width, grid_height) == (1024, 2304)
    assert estimate.image_files == 1
    assert estimate.bytes == storage_estimation.estimate_image_file_bytes(grid_width, grid_height, "png")


def test_estimates_img2img_extra_outputs():
    context = storage_estimation.GenerationStorageContext(width=256, height=256, batch_size=2, batch_count=2, tabname="img2img", has_mask=True)
    options = storage_estimation.StorageSavingOptions(
        samples_save=True,
        grid_save=False,
        save_init_img=True,
        save_images_before_color_correction=True,
        save_mask=True,
        save_mask_composite=True,
    )

    estimate = storage_estimation.estimate_generation_storage(context, options)

    assert estimate.image_files == 17
    assert estimate.text_files == 0


def test_format_bytes_uses_mb_and_gb():
    assert storage_estimation.format_bytes(1536 * 1024) == "1.5 MB"
    assert storage_estimation.format_bytes(1536 * 1024 * 1024) == "1.5 GB"
