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

    def ui(self, is_img2img):
        if not is_img2img:
            return None

        target_height = gr.Slider(label="Target height", minimum=64, maximum=4096, step=64, value=2048, elem_id=self.elem_id("target_height"))
        scale = gr.Slider(label="Scale", minimum=0.0, maximum=1.0, step=0.05, value=0.0, elem_id=self.elem_id("scale"))
        placement = gr.Dropdown(label="Placement", choices=PLACEMENT_CHOICES, value="Center", elem_id=self.elem_id("placement"))
        seam_size = gr.Slider(label="Seam size", minimum=0, maximum=128, step=8, value=32, elem_id=self.elem_id("seam_size"))
        mask_blur = gr.Slider(label="Mask blur", minimum=0, maximum=64, step=1, value=4, elem_id=self.elem_id("mask_blur"))
        inpainting_fill = gr.Radio(label="Masked content", choices=MASKED_CONTENT_CHOICES, value="fill", type="index", elem_id=self.elem_id("inpainting_fill"))
        inject_continue_prompt = gr.Checkbox(label="Inject continue prompt", value=False, elem_id=self.elem_id("inject_continue_prompt"))
        continue_prompt = gr.Textbox(label="Continue prompt", value=DEFAULT_CONTINUE_PROMPT, lines=2, elem_id=self.elem_id("continue_prompt"))

        return [target_height, scale, placement, seam_size, mask_blur, inpainting_fill, inject_continue_prompt, continue_prompt]

    def run(self, p, target_height, scale, placement, seam_size, mask_blur, inpainting_fill, inject_continue_prompt, continue_prompt):
        if not getattr(p, "init_images", None):
            return Processed(p, [], p.seed, "Vertical Outpainting Mk5 requires one source image.")

        source = p.init_images[0].convert("RGB")
        target_height = int(target_height)
        if target_height < source.height:
            return Processed(p, [], p.seed, "Target height must be greater than or equal to the source image height.")

        prepared_source = source if float(scale) == 1.0 else prepare_scaled_source(source, target_height, scale)
        source_top = compute_vertical_offset(prepared_source.height, target_height, placement)
        target_canvas = create_target_canvas(prepared_source, target_height, source_top)
        mask = build_outpaint_mask(prepared_source.size, target_height, source_top, seam_size)

        original_prompt = p.prompt
        original_init_images = p.init_images
        original_image_mask = getattr(p, "image_mask", None)
        original_width = p.width
        original_height = p.height
        original_n_iter = p.n_iter
        original_batch_size = p.batch_size
        original_do_not_save_grid = p.do_not_save_grid
        original_do_not_save_samples = p.do_not_save_samples
        original_mask_blur = p.mask_blur
        original_inpainting_fill = p.inpainting_fill
        has_inpainting_mask_invert = hasattr(p, "inpainting_mask_invert")
        original_inpainting_mask_invert = getattr(p, "inpainting_mask_invert", None)
        has_inpaint_full_res = hasattr(p, "inpaint_full_res")
        original_inpaint_full_res = getattr(p, "inpaint_full_res", None)

        p.extra_generation_params["Vertical Outpainting MK5 target height"] = target_height
        p.extra_generation_params["Vertical Outpainting MK5 scale"] = float(scale)
        p.extra_generation_params["Vertical Outpainting MK5 placement"] = placement
        p.extra_generation_params["Vertical Outpainting MK5 seam size"] = int(seam_size)
        p.extra_generation_params["Vertical Outpainting MK5 mask blur"] = int(mask_blur)
        p.extra_generation_params["Vertical Outpainting MK5 continue prompt injected"] = bool(inject_continue_prompt)

        try:
            if float(scale) == 1.0 or not mask_has_white(mask):
                final_image = target_canvas
                if should_save_final_samples(opts, state, original_do_not_save_samples):
                    images.save_image(final_image, p.outpath_samples, "", p.seed, original_prompt, opts.samples_format, info="", p=p)
                return Processed(p, [final_image], p.seed, "", all_seeds=[p.seed], infotexts=[""])

            if inject_continue_prompt:
                p.prompt = _append_continue_prompt(p.prompt, continue_prompt)

            p.n_iter = 1
            p.batch_size = 1
            p.do_not_save_grid = True
            p.do_not_save_samples = True
            p.width = source.width
            p.height = target_height
            p.mask_blur = int(mask_blur)
            p.inpainting_fill = inpainting_fill
            p.inpainting_mask_invert = 0
            state.job_count = 1
            state.job = "Vertical Outpainting Mk5: final canvas"

            p.init_images = [target_canvas]
            p.image_mask = mask
            processed = process_images(p)

            if not processed.images:
                return Processed(p, [], p.seed, "Vertical Outpainting Mk5 generation returned no images.", all_seeds=[], infotexts=[])

            final_image = restore_prepared_source(processed.images[0], prepared_source, source_top)
            if should_save_final_samples(opts, state, original_do_not_save_samples):
                images.save_image(final_image, p.outpath_samples, "", processed.seed, original_prompt, opts.samples_format, info=processed.info, p=p)

            return Processed(p, [final_image], processed.seed, processed.info, all_seeds=[processed.seed], infotexts=[processed.info])
        finally:
            p.prompt = original_prompt
            p.init_images = original_init_images
            p.image_mask = original_image_mask
            p.width = original_width
            p.height = original_height
            p.n_iter = original_n_iter
            p.batch_size = original_batch_size
            p.do_not_save_grid = original_do_not_save_grid
            p.do_not_save_samples = original_do_not_save_samples
            p.mask_blur = original_mask_blur
            p.inpainting_fill = original_inpainting_fill
            if has_inpainting_mask_invert:
                p.inpainting_mask_invert = original_inpainting_mask_invert
            elif hasattr(p, "inpainting_mask_invert"):
                delattr(p, "inpainting_mask_invert")
            if has_inpaint_full_res:
                p.inpaint_full_res = original_inpaint_full_res
