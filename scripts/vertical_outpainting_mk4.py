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

PLACEMENT_CHOICES = ["All", "Centered", "Top", "Bottom", "Top biased", "Bottom biased"]
SOURCE_HANDLING_CHOICES = ["Preserve source pixels", "Blend seam only", "Soft resynthesis"]
MASKED_CONTENT_CHOICES = ["fill", "original", "latent noise", "latent nothing"]
SOFT_RESYNTHESIS_INPAINTING_FILL = MASKED_CONTENT_CHOICES.index("original")


def _placement_offset(source_height, target_height, placement):
    gap = target_height - source_height
    if placement == "Centered":
        return gap // 2
    if placement == "Top":
        return 0
    if placement == "Bottom":
        return gap
    if placement == "Top biased":
        return gap // 4
    if placement == "Bottom biased":
        return (gap * 3) // 4
    raise ValueError(f"Unknown source placement: {placement}")


def build_placement_offsets(source_height, target_height, placement):
    source_height = int(source_height)
    target_height = int(target_height)
    if target_height < source_height:
        raise ValueError("Target height must be greater than or equal to the source image height.")

    names = PLACEMENT_CHOICES[1:] if placement == "All" else [placement]
    offsets = []
    seen_offsets = set()
    for name in names:
        offset = _placement_offset(source_height, target_height, name)
        if offset in seen_offsets:
            continue
        seen_offsets.add(offset)
        offsets.append((name, offset))
    return offsets


def create_target_canvas(source, target_height, source_top):
    canvas = Image.new("RGB", (source.width, int(target_height)), "black")
    canvas.paste(source.convert("RGB"), (0, int(source_top)))
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


def build_source_mask(source_size, target_height, source_top, source_handling, seam_size):
    source_width, source_height = source_size
    target_height = int(target_height)
    source_top = int(source_top)
    source_bottom = source_top + source_height

    if source_handling == "Soft resynthesis":
        return Image.new("L", (source_width, target_height), "white")

    seam = _clamped_seam(source_height, seam_size)
    has_top_gap = source_top > 0
    has_bottom_gap = source_bottom < target_height
    mask = Image.new("L", (source_width, target_height), "black")
    draw = ImageDraw.Draw(mask)

    if has_top_gap:
        draw.rectangle((0, 0, source_width, min(target_height, source_top + seam) - 1), fill="white")
    if has_bottom_gap:
        draw.rectangle((0, max(0, source_bottom - seam), source_width, target_height - 1), fill="white")

    return mask


def restore_source_region(generated, source, source_top, source_handling, seam_size):
    result = generated.copy()
    source = source.convert("RGB")
    source_top = int(source_top)

    if source_handling == "Soft resynthesis":
        return result

    if source_handling == "Preserve source pixels":
        result.paste(source, (0, source_top))
        return result

    if source_handling != "Blend seam only":
        raise ValueError(f"Unknown source handling: {source_handling}")

    target_height = result.height
    source_bottom = source_top + source.height
    seam = _clamped_seam(source.height, seam_size)
    crop_top = seam if source_top > 0 else 0
    crop_bottom = source.height - seam if source_bottom < target_height else source.height

    if crop_bottom > crop_top:
        protected = source.crop((0, crop_top, source.width, crop_bottom))
        result.paste(protected, (0, source_top + crop_top))

    return result


class Script(scripts.Script):
    def title(self):
        return "Vertical Outpainting Mk4"

    def show(self, is_img2img):
        return is_img2img
