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
