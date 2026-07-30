# Vertical Outpainting Mk4 Design

## Goal

Create a new img2img script named `Vertical Outpainting Mk4` that takes a source image such as `1024x1280` and generates a final image at a different vertical canvas size such as `1024x2048`, while keeping the source composition as the central reference.

Mk4 is not an iterative pass-based outpainter like Mk3. It is a target-canvas resynthesis script: it builds the final canvas first, places the source image into that canvas, creates a vertical inpaint mask, and runs generation at the final dimensions.

## Research Summary

Diffusion inpainting workflows treat the mask as the boundary between preserved context and generated content: masked regions are filled while unmasked regions provide image context. This makes a canvas-first vertical resize script a natural fit for img2img inpainting.

Reference-conditioning systems such as ControlNet and IP-Adapter can improve structural or identity consistency, but they require optional extensions or model support. The local installation does not currently include ControlNet or IP-Adapter extensions, so Mk4 must work without them.

Sources:

- Hugging Face Diffusers inpainting guide: https://huggingface.co/docs/diffusers/en/using-diffusers/inpaint
- ControlNet paper: https://arxiv.org/abs/2302.05543
- IP-Adapter paper: https://arxiv.org/abs/2308.06721

## Scope

This is a new script, not a modification of `scripts/vertical_outpainting_mk3.py`, `scripts/poor_mans_outpainting.py`, or `scripts/outpainting_mk_2.py`.

The first version supports vertical target-canvas generation only. Horizontal resizing, grids, external reference-extension integrations, and automatic ControlNet/IP-Adapter wiring are out of scope.

## User Interface

The script appears only in img2img and is titled `Vertical Outpainting Mk4`.

Controls:

- `Target height`: slider or number input from `64` to `4096`, step `64`, default `2048`.
- `Source placement`: dropdown with `All`, `Centered`, `Top`, `Bottom`, `Top biased`, and `Bottom biased`, defaulting to `All`.
- `Source handling`: dropdown with `Preserve source pixels`, `Blend seam only`, and `Soft resynthesis`, defaulting to `Preserve source pixels`.
- `Seam size`: slider from `0` to `128`, step `8`, default `32`.
- `Mask blur`: slider from `0` to `64`, step `1`, default `4`, matching the existing outpainting scripts.
- `Masked content`: radio matching `Poor man's outpainting` choices and behavior.
- `Inject continue prompt`: checkbox, default off.
- `Continue prompt`: textbox defaulting to `extend the image naturally, continue the existing scene, preserve composition, lighting, perspective, colors, style, and subject consistency`.

The output gallery contains only completed final images. The script does not generate or save a grid.

The script does not add new controls for denoising strength, sampler, CFG scale, batch size, or batch count. Those remain controlled by the existing img2img UI. Internally, Mk4 runs one placement at a time and returns the final placement results.

## Placement Semantics

The target width is always the source image width. The target height comes from `Target height`.

If `Target height` is equal to the source height, all placements are equivalent and the script returns one final image for the selected mode.

If `Target height` is smaller than the source height, the script returns `Processed(p, [], p.seed, "Target height must be greater than or equal to the source image height.")`. Mk4 is for vertical canvas expansion, not cropping.

For source height `H`, target height `T`, and total vertical gap `G = T - H`:

- `Centered`: source top offset is `G / 2`.
- `Top`: source top offset is `0`.
- `Bottom`: source top offset is `G`.
- `Top biased`: source top offset is `G / 4`.
- `Bottom biased`: source top offset is `3G / 4`.

Offsets are rounded to integers. The bottom gap receives the remaining pixels so the final canvas is exactly `target_width x target_height`.

When `Source placement` is `All`, the script runs all distinct placements in this order: `Centered`, `Top`, `Bottom`, `Top biased`, `Bottom biased`. Duplicate placements caused by small or zero gaps are collapsed by offset while preserving the first occurrence.

## Source Handling Modes

`Preserve source pixels` locks the original source region. The inpaint mask covers only the new top and bottom gaps plus the configured seam band. After generation, the original source image is pasted back over its placement region exactly. This mode prioritizes identity and detail preservation.

`Blend seam only` locks most of the source image but allows the seam band around the source edges to regenerate. After generation, the script pastes back only the protected inner source rectangle, leaving the seam band from the generated result. This mode prioritizes hiding visible borders while preserving most source pixels. If a source edge does not touch a generated gap, that edge has no seam band and remains part of the protected source region.

`Soft resynthesis` uses the whole target canvas as a generation surface while still placing the source image as the init image context. The script does not paste the source image back afterward. This mode prioritizes a unified final image, but source details may drift. For this mode, the internal masked content value is forced to `original` so the source-containing target canvas remains the image signal.

## Processing Model

For each selected placement:

1. Create an RGB target canvas with the source width and selected target height.
2. Paste the source image into the target canvas at the placement offset.
3. Build a grayscale mask according to the selected source handling mode.
4. Optionally append the continue prompt to `p.prompt` for the internal generation call.
5. Configure img2img/inpainting state for one final-size pass:
   - `p.init_images = [target_canvas]`
   - `p.image_mask = mask`
   - `p.width = target_width`
   - `p.height = target_height`
   - `p.mask_blur = Mask blur`
   - `p.inpainting_fill = Masked content`, except `Soft resynthesis`, which uses `original`
   - `p.n_iter = 1`
   - `p.batch_size = 1`
   - `p.do_not_save_grid = True`
6. Run `process_images(p)`.
7. Apply post-generation source restoration if required by the selected source handling mode.
8. Save and return the completed final image.

The script restores the original processing object fields after each internal run so later scripts and runs are not polluted by Mk4 state. Restored fields include prompt, init images, mask, width, height, mask blur, inpainting fill, batch settings, and grid/sample saving flags.

## Mask Details

For `Preserve source pixels`, the mask is white in the new top and bottom canvas areas. The mask also includes a seam band of `Seam size` pixels extending into the source region at each generated edge. The post-generation paste restores the entire source image exactly.

For `Blend seam only`, the mask is white in the new top and bottom canvas areas and in the seam band. The post-generation paste restores only the source area excluding the seam bands that touched generated regions.

For `Soft resynthesis`, the mask is white across the full target canvas and the internal masked content is `original`. This makes the source image an init/reference layer instead of an exact pixel lock.

If only one side has a gap for a placement, only that side receives a seam band. If both sides have gaps, both sides receive seam bands. Seam bands are clamped so they never exceed the source image height.

## Seed And Batch Behavior

Each final placement variant is generated independently from the same source image and the same user settings. The first generated variant uses the incoming seed. Each subsequent variant advances the seed by one from the previous processed result, matching the sequential style used by Mk3.

The first version processes one init image per script invocation. Output count equals the number of selected distinct placements.

## Saving And Metadata

If sample saving is enabled, each final placement result is saved as an individual sample. The script does not save a grid.

Generation metadata must include script-specific parameters in `p.extra_generation_params`, including:

- `Vertical Outpainting MK4 target height`
- `Vertical Outpainting MK4 source placement`
- `Vertical Outpainting MK4 source handling`
- `Vertical Outpainting MK4 seam size`
- `Vertical Outpainting MK4 mask blur`
- `Vertical Outpainting MK4 continue prompt injected`

## Error Handling

The script returns an empty `Processed` result with a clear message when:

- The source image is missing.
- `Target height` is smaller than the source image height.
- No valid placement remains after duplicate placement collapse.

## Testing

The implementation must include focused helper-level tests for:

- Building distinct placement offsets for centered, top, bottom, and biased placements.
- Rejecting target heights smaller than the source height.
- Building mask regions for all three source handling modes.
- Restoring exact source pixels in `Preserve source pixels`.
- Preserving only the protected inner source rectangle in `Blend seam only`.
- Collapsing duplicate placements when the target height equals the source height.

Runtime verification must include importing the new script and running Python compilation for the new file.
