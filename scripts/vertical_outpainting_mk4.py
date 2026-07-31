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


def soft_blend_vertical_seams(generated, source, source_top, blend_width):
    result = generated.copy().convert("RGB")
    source = source.convert("RGB")
    source_top = int(source_top)
    blend_width = max(0, min(int(blend_width), source.height, result.height))
    if blend_width <= 0:
        result.paste(source, (0, source_top))
        return result

    result.paste(source, (0, source_top))

    return soft_blend_vertical_seam_bands(result, generated, source, source_top, blend_width)


def soft_blend_vertical_seam_bands(result, generated, source, source_top, blend_width):
    source = source.convert("RGB")
    source_top = int(source_top)
    blend_width = max(0, min(int(blend_width), source.height, result.height))
    if blend_width <= 0:
        return result

    source_bottom = source_top + source.height
    top_width = 0
    bottom_width = 0

    if source_top > 0:
        top_width = min(blend_width, source.height, source_top, result.height - source_top)
    if source_bottom < result.height:
        bottom_width = min(blend_width, source.height, source_bottom, result.height - source_bottom)

    if top_width + bottom_width > source.height:
        top_width = min(top_width, (source.height + 1) // 2)
        bottom_width = min(bottom_width, source.height // 2)

    if top_width:
        for index in range(top_width):
            alpha = (index + 1) / (top_width + 1)
            generated_row = generated.crop((0, source_top + index, source.width, source_top + index + 1)).convert("RGB")
            source_row = source.crop((0, index, source.width, index + 1))
            result.paste(Image.blend(generated_row, source_row, alpha), (0, source_top + index))

    if bottom_width:
        for index in range(bottom_width):
            alpha = 1.0 - ((index + 1) / (bottom_width + 1))
            source_y = source.height - bottom_width + index
            result_y = source_bottom - bottom_width + index
            generated_row = generated.crop((0, result_y, source.width, result_y + 1)).convert("RGB")
            source_row = source.crop((0, source_y, source.width, source_y + 1))
            result.paste(Image.blend(generated_row, source_row, alpha), (0, result_y))

    return result


def restore_source_region(generated, source, source_top, source_handling, seam_size, soft_seam_blend=0):
    result = generated.copy()
    source = source.convert("RGB")
    source_top = int(source_top)

    if source_handling == "Soft resynthesis":
        return result

    if source_handling == "Preserve source pixels":
        if int(soft_seam_blend) > 0:
            return soft_blend_vertical_seams(generated, source, source_top, soft_seam_blend)
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

    if int(soft_seam_blend) > 0:
        result = soft_blend_vertical_seam_bands(result, generated, source, source_top, soft_seam_blend)

    return result


class Script(scripts.Script):
    def title(self):
        return "Vertical Outpainting Mk4"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        if not is_img2img:
            return None

        target_height = gr.Slider(label="Target height", minimum=64, maximum=4096, step=64, value=2048, elem_id=self.elem_id("target_height"))
        source_placement = gr.Dropdown(label="Source placement", choices=PLACEMENT_CHOICES, value="All", elem_id=self.elem_id("source_placement"))
        source_handling = gr.Dropdown(label="Source handling", choices=SOURCE_HANDLING_CHOICES, value="Preserve source pixels", elem_id=self.elem_id("source_handling"))
        seam_size = gr.Slider(label="Seam size", minimum=0, maximum=128, step=8, value=32, elem_id=self.elem_id("seam_size"))
        mask_blur = gr.Slider(label="Mask blur", minimum=0, maximum=64, step=1, value=4, elem_id=self.elem_id("mask_blur"))
        soft_seam_blend = gr.Slider(label="Soft seam blend", minimum=0, maximum=128, step=8, value=32, elem_id=self.elem_id("soft_seam_blend"))
        inpainting_fill = gr.Radio(label="Masked content", choices=MASKED_CONTENT_CHOICES, value="fill", type="index", elem_id=self.elem_id("inpainting_fill"))
        inject_continue_prompt = gr.Checkbox(label="Inject continue prompt", value=False, elem_id=self.elem_id("inject_continue_prompt"))
        continue_prompt = gr.Textbox(label="Continue prompt", value=DEFAULT_CONTINUE_PROMPT, lines=2, elem_id=self.elem_id("continue_prompt"))
        variants_per_input = gr.Slider(label="Variants per input image", minimum=1, maximum=100, step=1, value=1, elem_id=self.elem_id("variants_per_input"))

        return [target_height, source_placement, source_handling, seam_size, mask_blur, soft_seam_blend, inpainting_fill, inject_continue_prompt, continue_prompt, variants_per_input]

    def run(self, p, target_height, source_placement, source_handling, seam_size, mask_blur, soft_seam_blend, inpainting_fill, inject_continue_prompt, continue_prompt, variants_per_input=1):
        if not getattr(p, "init_images", None):
            return Processed(p, [], p.seed, "Vertical Outpainting Mk4 requires one source image.")

        source = p.init_images[0].convert("RGB")
        target_height = int(target_height)
        try:
            placements = build_placement_offsets(source.height, target_height, source_placement)
        except ValueError as exc:
            return Processed(p, [], p.seed, str(exc))

        if not placements:
            return Processed(p, [], p.seed, "No valid source placement selected.")

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

        p.extra_generation_params["Vertical Outpainting MK4 target height"] = target_height
        p.extra_generation_params["Vertical Outpainting MK4 source placement"] = source_placement
        p.extra_generation_params["Vertical Outpainting MK4 source handling"] = source_handling
        p.extra_generation_params["Vertical Outpainting MK4 seam size"] = int(seam_size)
        p.extra_generation_params["Vertical Outpainting MK4 mask blur"] = int(mask_blur)
        p.extra_generation_params["Vertical Outpainting MK4 soft seam blend"] = int(soft_seam_blend)
        p.extra_generation_params["Vertical Outpainting MK4 continue prompt injected"] = bool(inject_continue_prompt)
        variants_per_input = max(1, int(variants_per_input))
        p.extra_generation_params["Vertical Outpainting MK4 variants per input image"] = variants_per_input

        final_images = []
        variant_seeds = []
        variant_infos = []

        try:
            if inject_continue_prompt:
                p.prompt = _append_continue_prompt(p.prompt, continue_prompt)

            p.n_iter = 1
            p.batch_size = 1
            p.do_not_save_grid = True
            p.do_not_save_samples = True
            p.width = source.width
            p.height = target_height
            p.mask_blur = int(mask_blur)
            p.inpainting_fill = SOFT_RESYNTHESIS_INPAINTING_FILL if source_handling == "Soft resynthesis" else inpainting_fill
            p.inpainting_mask_invert = 0
            state.job_count = len(placements) * variants_per_input

            for index, (placement_name, source_top) in enumerate(placements):
                for variant_index in range(variants_per_input):
                    state.job = f"{placement_name}: variant {variant_index + 1} out of {variants_per_input}, final canvas {index + 1} out of {len(placements)}"
                    target_canvas = create_target_canvas(source, target_height, source_top)
                    mask = build_source_mask(source.size, target_height, source_top, source_handling, seam_size)

                    p.init_images = [target_canvas]
                    p.image_mask = mask
                    processed = process_images(p)

                    if not processed.images:
                        break

                    generated = processed.images[0]
                    final_image = restore_source_region(generated, source, source_top, source_handling, seam_size, soft_seam_blend)
                    final_images.append(final_image)
                    variant_seeds.append(processed.seed)
                    variant_infos.append(processed.info)
                    p.seed = processed.seed + 1
                else:
                    continue
                break

            should_save_samples = (
                opts.samples_save
                and not original_do_not_save_samples
                and (
                    getattr(opts, "save_incomplete_images", False)
                    or not getattr(state, "interrupted", False) and not getattr(state, "skipped", False)
                )
            )
            if should_save_samples:
                for image, seed, info in zip(final_images, variant_seeds, variant_infos):
                    images.save_image(image, p.outpath_samples, "", seed, original_prompt, opts.samples_format, info=info, p=p)

            return Processed(
                p,
                final_images,
                variant_seeds[0] if variant_seeds else p.seed,
                variant_infos[0] if variant_infos else "",
                all_seeds=variant_seeds,
                infotexts=variant_infos,
            )
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
            else:
                delattr(p, "inpainting_mask_invert")
            if has_inpaint_full_res:
                p.inpaint_full_res = original_inpaint_full_res
