import math

import gradio as gr
import modules.scripts as scripts
from PIL import Image, ImageDraw

from modules import devices, images
from modules.processing import Processed, process_images
from modules.shared import opts, state


DEFAULT_CONTINUE_PROMPT = (
    "extend the image naturally, continue the existing scene, preserve composition, "
    "lighting, perspective, colors, style, and subject consistency"
)

SHIFT_PRESET_CHOICES = [
    "All",
    "Centered",
    "Top",
    "Bottom",
    "Top Centered",
    "Bottom Centered",
]


def _preset_sequences(pixels):
    half_pixels = pixels // 2
    return {
        "Centered": [("up", pixels), ("down", pixels)],
        "Top": [("up", pixels), ("up", pixels)],
        "Bottom": [("down", pixels), ("down", pixels)],
        "Top Centered": [("up", pixels), ("down", half_pixels), ("up", half_pixels)],
        "Bottom Centered": [("down", pixels), ("up", half_pixels), ("down", half_pixels)],
    }


def _is_sequence_allowed(sequence, selected_directions):
    return all(direction in selected_directions for direction, _ in sequence)


def build_shift_sequences(direction, shift_preset, pixels):
    selected_directions = set(direction or [])
    sequences = _preset_sequences(int(pixels))

    if shift_preset == "All":
        names = ["Centered", "Top", "Bottom", "Top Centered", "Bottom Centered"]
    else:
        names = [shift_preset]

    return [
        (name, sequences[name])
        for name in names
        if name in sequences and _is_sequence_allowed(sequences[name], selected_directions)
    ]


def zoom_content_centered(image, zoom):
    zoom = float(zoom)
    if zoom <= 1.0:
        return image.copy()

    width, height = image.size
    resized_width = max(width, int(round(width * zoom)))
    resized_height = max(height, int(round(height * zoom)))
    resized = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

    left = (resized_width - width) // 2
    top = (resized_height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def _append_continue_prompt(prompt, continue_prompt):
    continue_prompt = (continue_prompt or "").strip()
    if not continue_prompt:
        return prompt
    if not prompt:
        return continue_prompt
    return f"{prompt}, {continue_prompt}"


def _round_up_to_64(value):
    return math.ceil(value / 64) * 64


def _expand_vertical_once(p, init_img, expand_pixels, pass_direction, mask_blur):
    initial_seed = None
    initial_info = None
    up = int(expand_pixels) if pass_direction == "up" else 0
    down = int(expand_pixels) if pass_direction == "down" else 0

    target_w = _round_up_to_64(init_img.width)
    target_h = _round_up_to_64(init_img.height + up + down)

    if up > 0:
        up = up * (target_h - init_img.height) // (up + down)
    if down > 0:
        down = target_h - init_img.height - up

    img = Image.new("RGB", (target_w, target_h))
    img.paste(init_img, ((target_w - init_img.width) // 2, up))

    mask = Image.new("L", (img.width, img.height), "white")
    draw = ImageDraw.Draw(mask)
    draw.rectangle((
        0,
        up + (mask_blur * 2 if up > 0 else 0),
        mask.width,
        mask.height - down - (mask_blur * 2 if down > 0 else 0),
    ), fill="black")

    latent_mask = Image.new("L", (img.width, img.height), "white")
    latent_draw = ImageDraw.Draw(latent_mask)
    latent_draw.rectangle((
        0,
        up + (mask_blur // 2 if up > 0 else 0),
        latent_mask.width,
        latent_mask.height - down - (mask_blur // 2 if down > 0 else 0),
    ), fill="black")

    devices.torch_gc()

    grid = images.split_grid(img, tile_w=p.width, tile_h=p.height, overlap=expand_pixels)
    grid_mask = images.split_grid(mask, tile_w=p.width, tile_h=p.height, overlap=expand_pixels)
    grid_latent_mask = images.split_grid(latent_mask, tile_w=p.width, tile_h=p.height, overlap=expand_pixels)

    work = []
    work_mask = []
    work_latent_mask = []
    work_results = []

    for (y, h, row), (_, _, row_mask), (_, _, row_latent_mask) in zip(grid.tiles, grid_mask.tiles, grid_latent_mask.tiles):
        for tiledata, tiledata_mask, tiledata_latent_mask in zip(row, row_mask, row_latent_mask):
            tile_inside_original_vertical = y >= up and y + h <= img.height - down
            if tile_inside_original_vertical:
                continue

            work.append(tiledata[2])
            work_mask.append(tiledata_mask[2])
            work_latent_mask.append(tiledata_latent_mask[2])

    print(f"Vertical Outpainting Mk3 will process {len(work)} tiles for {pass_direction} expansion.")

    for i in range(len(work)):
        p.init_images = [work[i]]
        p.image_mask = work_mask[i]
        p.latent_mask = work_latent_mask[i]
        state.job = f"Outpainting {pass_direction} tile {i + 1} out of {len(work)}"

        processed = process_images(p)

        if initial_seed is None:
            initial_seed = processed.seed
            initial_info = processed.info

        p.seed = processed.seed + 1
        work_results += processed.images

    image_index = 0
    for y, h, row in grid.tiles:
        for tiledata in row:
            tile_inside_original_vertical = y >= up and y + h <= img.height - down
            if tile_inside_original_vertical:
                continue

            tiledata[2] = work_results[image_index] if image_index < len(work_results) else Image.new("RGB", (p.width, p.height))
            image_index += 1

    combined_image = images.combine_grid(grid)
    return combined_image.crop((0, 0, target_w, init_img.height + up + down)), initial_seed, initial_info


class Script(scripts.Script):
    def title(self):
        return "Vertical Outpainting Mk3"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        if not is_img2img:
            return None

        pixels = gr.Slider(label="Pixels to expand", minimum=8, maximum=384, step=8, value=384, elem_id=self.elem_id("pixels"))
        mask_blur = gr.Slider(label="Mask blur", minimum=0, maximum=64, step=1, value=4, elem_id=self.elem_id("mask_blur"))
        inpainting_fill = gr.Radio(label="Masked content", choices=["fill", "original", "latent noise", "latent nothing"], value="fill", type="index", elem_id=self.elem_id("inpainting_fill"))
        direction = gr.CheckboxGroup(label="Outpainting direction", choices=["up", "down"], value=["up", "down"], elem_id=self.elem_id("direction"))
        shift_preset = gr.Dropdown(label="Shift preset", choices=SHIFT_PRESET_CHOICES, value="All", elem_id=self.elem_id("shift_preset"))
        zoom = gr.Slider(label="Zoom in", minimum=1.0, maximum=3.0, step=0.05, value=1.0, elem_id=self.elem_id("zoom"))
        inject_continue_prompt = gr.Checkbox(label="Inject continue prompt", value=False, elem_id=self.elem_id("inject_continue_prompt"))
        continue_prompt = gr.Textbox(label="Continue prompt", value=DEFAULT_CONTINUE_PROMPT, lines=2, elem_id=self.elem_id("continue_prompt"))

        return [pixels, mask_blur, inpainting_fill, direction, shift_preset, zoom, inject_continue_prompt, continue_prompt]

    def run(self, p, pixels, mask_blur, inpainting_fill, direction, shift_preset, zoom, inject_continue_prompt, continue_prompt):
        original_prompt = p.prompt
        original_init_images = p.init_images
        original_n_iter = p.n_iter
        original_batch_size = p.batch_size
        original_do_not_save_grid = p.do_not_save_grid
        original_do_not_save_samples = p.do_not_save_samples
        original_mask_blur = p.mask_blur
        original_inpainting_fill = p.inpainting_fill
        original_inpaint_full_res = p.inpaint_full_res

        sequences = build_shift_sequences(direction, shift_preset, int(pixels))
        if not sequences:
            return Processed(p, [], p.seed, "No vertical outpainting direction selected.")

        p.extra_generation_params["Vertical Outpainting MK3 pixels"] = int(pixels)
        p.extra_generation_params["Vertical Outpainting MK3 directions"] = ", ".join(direction or [])
        p.extra_generation_params["Vertical Outpainting MK3 shift preset"] = shift_preset
        p.extra_generation_params["Vertical Outpainting MK3 zoom"] = float(zoom)
        p.extra_generation_params["Vertical Outpainting MK3 inject continue prompt"] = bool(inject_continue_prompt)

        try:
            if inject_continue_prompt:
                p.prompt = _append_continue_prompt(p.prompt, continue_prompt)

            p.mask_blur = mask_blur * 2
            p.inpainting_fill = inpainting_fill
            p.inpaint_full_res = False
            p.n_iter = 1
            p.batch_size = 1
            p.do_not_save_grid = True
            p.do_not_save_samples = True

            prepared_image = zoom_content_centered(original_init_images[0], zoom)
            state.job_count = sum(len(passes) for _, passes in sequences)

            final_images = []
            initial_seed = None
            initial_info = None

            for preset_name, passes in sequences:
                current_image = prepared_image.copy()
                for pass_index, (pass_direction, pass_pixels) in enumerate(passes):
                    state.job = f"{preset_name}: pass {pass_index + 1} out of {len(passes)}"
                    current_image, pass_seed, pass_info = _expand_vertical_once(p, current_image, pass_pixels, pass_direction, mask_blur)
                    if initial_seed is None and pass_seed is not None:
                        initial_seed = pass_seed
                        initial_info = pass_info

                final_images.append(current_image)

            if opts.samples_save:
                for image in final_images:
                    images.save_image(image, p.outpath_samples, "", initial_seed, original_prompt, opts.samples_format, info=initial_info, p=p)

            return Processed(p, final_images, initial_seed, initial_info)
        finally:
            p.prompt = original_prompt
            p.init_images = original_init_images
            p.n_iter = original_n_iter
            p.batch_size = original_batch_size
            p.do_not_save_grid = original_do_not_save_grid
            p.do_not_save_samples = original_do_not_save_samples
            p.mask_blur = original_mask_blur
            p.inpainting_fill = original_inpainting_fill
            p.inpaint_full_res = original_inpaint_full_res
