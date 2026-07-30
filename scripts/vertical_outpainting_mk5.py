import gradio as gr
import modules.scripts as scripts
from PIL import Image, ImageDraw

from modules import images
from modules.processing import Processed, process_images
from modules.shared import opts, state


DEFAULT_CONTINUE_PROMPT = (
    "extend the image naturally, continue the existing scene, preserve composition, "
    "lighting, perspective, colors, style, and subject consistency"
)

PLACEMENT_CHOICES = ["Center", "Top", "Bottom"]
MASKED_CONTENT_CHOICES = ["fill", "original", "latent noise", "latent nothing"]


def compute_scale_factor(source_height, target_height, scale):
    source_height = int(source_height)
    target_height = int(target_height)
    scale = max(0.0, min(float(scale), 1.0))
    return 1.0 + scale * ((target_height / source_height) - 1.0)


def prepare_scaled_source(source, target_height, scale):
    source = source.convert("RGB")
    scale_factor = compute_scale_factor(source.height, target_height, scale)
    scaled_width = max(source.width, int(round(source.width * scale_factor)))
    scaled_height = min(int(target_height), max(1, int(round(source.height * scale_factor))))
    resized = source.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
    left = max(0, (scaled_width - source.width) // 2)
    return resized.crop((left, 0, left + source.width, scaled_height))


def compute_vertical_offset(source_height, target_height, placement):
    gap = int(target_height) - int(source_height)
    if gap < 0:
        raise ValueError("Target height must be greater than or equal to the source image height.")
    if placement == "Top":
        return 0
    if placement == "Bottom":
        return gap
    if placement == "Center":
        return gap // 2
    raise ValueError(f"Unknown placement: {placement}")


def create_target_canvas(prepared_source, target_height, source_top):
    canvas = Image.new("RGB", (prepared_source.width, int(target_height)), "black")
    canvas.paste(prepared_source.convert("RGB"), (0, int(source_top)))
    return canvas


def _append_continue_prompt(prompt, continue_prompt):
    continue_prompt = (continue_prompt or "").strip()
    if not continue_prompt:
        return prompt
    if not prompt:
        return continue_prompt
    return f"{prompt}, {continue_prompt}"


def _clamped_seam(source_height, seam_size):
    return max(0, min(int(seam_size), int(source_height)))


def build_outpaint_mask(source_size, target_height, source_top, seam_size):
    source_width, source_height = source_size
    target_height = int(target_height)
    source_top = int(source_top)
    source_bottom = source_top + source_height
    seam = _clamped_seam(source_height, seam_size)
    mask = Image.new("L", (source_width, target_height), "black")
    draw = ImageDraw.Draw(mask)

    if source_top > 0:
        draw.rectangle((0, 0, source_width, min(target_height, source_top + seam) - 1), fill="white")
    if source_bottom < target_height:
        draw.rectangle((0, max(0, source_bottom - seam), source_width, target_height - 1), fill="white")

    return mask


def restore_prepared_source(generated, prepared_source, source_top):
    result = generated.copy()
    result.paste(prepared_source.convert("RGB"), (0, int(source_top)))
    return result


def mask_has_white(mask):
    return mask.getbbox() is not None


def should_save_final_samples(opts_obj, state_obj, original_do_not_save_samples):
    return (
        bool(getattr(opts_obj, "samples_save", False))
        and not bool(original_do_not_save_samples)
        and (
            bool(getattr(opts_obj, "save_incomplete_images", False))
            or not bool(getattr(state_obj, "interrupted", False)) and not bool(getattr(state_obj, "skipped", False))
        )
    )


class Script(scripts.Script):
    def title(self):
        return "Vertical Outpainting Mk5"

    def show(self, is_img2img):
        return is_img2img
