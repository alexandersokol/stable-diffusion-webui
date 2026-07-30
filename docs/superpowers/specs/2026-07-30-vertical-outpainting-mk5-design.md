# Vertical Outpainting Mk5 Design

## Goal

Create a new img2img script named `Vertical Outpainting Mk5` that turns a source image such as `1024x1280` into a taller target image such as `1024x2048` by scaling the source content inside the target canvas, anchoring it vertically, generating any remaining empty vertical areas, and restoring the scaled source pixels exactly.

Mk5 is a focused new script, not a mode added to Mk4.

## Scope

This is a new script, not a modification of `scripts/vertical_outpainting_mk3.py`, `scripts/vertical_outpainting_mk4.py`, `scripts/poor_mans_outpainting.py`, or `scripts/outpainting_mk_2.py`.

The first version supports vertical target-canvas generation only. Horizontal target resizing, placement variants, grids, zoom-out controls, ControlNet/IP-Adapter wiring, and soft source resynthesis are out of scope.

## User Interface

The script appears only in img2img and is titled `Vertical Outpainting Mk5`.

Controls:

- `Target height`: slider or number input from `64` to `4096`, step `64`, default `2048`.
- `Scale`: slider from `0.00` to `1.00`, step `0.05`, default `0.00`.
- `Placement`: dropdown with `Center`, `Top`, and `Bottom`, defaulting to `Center`.
- `Seam size`: slider from `0` to `128`, step `8`, default `32`.
- `Mask blur`: slider from `0` to `64`, step `1`, default `4`.
- `Masked content`: radio matching the existing outpainting choices: `fill`, `original`, `latent noise`, and `latent nothing`.
- `Inject continue prompt`: checkbox, default off.
- `Continue prompt`: textbox defaulting to `extend the image naturally, continue the existing scene, preserve composition, lighting, perspective, colors, style, and subject consistency`.

The output gallery contains only the completed final image. The script does not generate or save a grid.

The script does not add new controls for denoising strength, sampler, CFG scale, batch size, or batch count. Those remain controlled by the existing img2img UI.

## Geometry

The target width is always the source image width. The target height comes from `Target height`.

Mk5 requires `Target height` to be greater than or equal to the source image height. If the target height is smaller, the script returns `Processed(p, [], p.seed, "Target height must be greater than or equal to the source image height.")`. If the target height equals the source image height, the script follows the same no-gap path as `Scale = 1.00`: it returns the prepared source-only image without running generation.

For source width `W`, source height `H`, target height `T`, and user scale `S`:

- The scale at `S = 0.00` is `1.0`, so the source remains width-fit at `W x H`.
- The scale at `S = 1.00` is `T / H`, so the scaled source height touches the target height.
- Intermediate values linearly interpolate between those two factors:
  - `scale_factor = 1.0 + S * ((T / H) - 1.0)`.

The source is resized to:

- `scaled_width = round(W * scale_factor)`
- `scaled_height = round(H * scale_factor)`

The resized source is then horizontally center-cropped to the target width `W`. For the common `1024x1280` to `1024x2048` case, `S = 1.00` creates an intermediate `1638x2048` image and crops the horizontal center back to `1024x2048`.

Vertical placement determines where the scaled source sits inside the target height:

- `Top`: scaled source top is anchored at `0`; any empty area is below it.
- `Bottom`: scaled source bottom is anchored at `T`; any empty area is above it.
- `Center`: scaled source is vertically centered; empty area is split between top and bottom.

Because `Target height >= source height` and `S` is in `[0, 1]`, the scaled height is never intentionally larger than the target height. Rounding is clamped so the pasted/cropped source rectangle stays within the exact `W x T` target canvas.

## Masking And Source Preservation

Mk5 creates a final RGB target canvas of `W x T`, pastes the scaled/cropped source into it, and builds a grayscale mask.

The mask is white in empty vertical areas and in a `Seam size` band extending into the scaled source at each generated edge. The mask is black in the protected source area.

After generation, Mk5 pastes the scaled/cropped source image back into the target canvas exactly. This means generated pixels can blend into the seam during the inpaint pass, but the final source rectangle is exact and stable.

If `Scale = 1.00`, there are no empty vertical areas. The mask is blank, and the script returns the scaled/cropped source image without calling `process_images`.

If only one side has an empty area, only that side receives a seam band. If both sides have empty areas, both sides receive seam bands. Seam bands are clamped so they never exceed the scaled source height.

## Processing Model

For one selected input image:

1. Validate that a source image exists and that `Target height >= source height`.
2. Compute the scale factor from the `Scale` slider.
3. Resize the source image and horizontally center-crop it to the target width.
4. Compute the vertical placement offset from `Placement`.
5. Create the final target canvas and paste the prepared source at that offset.
6. Build the mask for empty areas plus seam bands.
7. If the mask has no white pixels, return the prepared target canvas as the only final image without running generation. This source-only result still records Mk5 metadata and is saved when sample saving is allowed.
8. Optionally append the continue prompt to `p.prompt` for the internal generation call.
9. Configure img2img/inpainting state for one final-size pass:
   - `p.init_images = [target_canvas]`
   - `p.image_mask = mask`
   - `p.width = target_width`
   - `p.height = target_height`
   - `p.mask_blur = Mask blur`
   - `p.inpainting_fill = Masked content`
   - `p.inpainting_mask_invert = 0`
   - `p.n_iter = 1`
   - `p.batch_size = 1`
   - `p.do_not_save_grid = True`
   - `p.do_not_save_samples = True`
10. Run `process_images(p)`.
11. If generation returns no images, return an empty `Processed` result and do not save a placeholder.
12. Paste the prepared source back exactly over the generated image.
13. Save and return the completed final image when sample saving is allowed.

The script restores the original processing object fields after the internal run so later scripts and runs are not polluted by Mk5 state. Restored fields include prompt, init images, mask, width, height, mask blur, inpainting fill, mask inversion, inpaint-full-res, batch settings, and grid/sample saving flags.

## Saving And Metadata

If sample saving is enabled and not suppressed by the original request, the final image is saved as an individual sample. The script does not save a grid.

Manual final saving must honor the same request and interruption policy used by normal processing:

- `opts.samples_save`
- the original `p.do_not_save_samples`
- `opts.save_incomplete_images`
- `state.interrupted`
- `state.skipped`

Generation metadata must include script-specific parameters in `p.extra_generation_params`, including:

- `Vertical Outpainting MK5 target height`
- `Vertical Outpainting MK5 scale`
- `Vertical Outpainting MK5 placement`
- `Vertical Outpainting MK5 seam size`
- `Vertical Outpainting MK5 mask blur`
- `Vertical Outpainting MK5 continue prompt injected`

## Error Handling

The script returns an empty `Processed` result with a clear message when:

- The source image is missing.
- `Target height` is smaller than the source image height.
- Generation returns no image for a non-empty mask.

## Testing

The implementation must include focused helper-level tests for:

- Computing scale factors at `0.00`, `0.50`, and `1.00`.
- Preparing a scaled/cropped source image while preserving the target width.
- Computing `Top`, `Bottom`, and `Center` vertical offsets.
- Building masks for top-only, bottom-only, centered, and no-gap cases.
- Returning the source-only target canvas without calling `process_images` when the mask is blank.
- Restoring exact prepared source pixels after generation.
- Forcing and restoring normal mask polarity.
- Honoring sample-save suppression and interrupted-generation save policy.
- Returning no placeholder image when generation returns an empty image list.

Runtime verification must include importing the new script, running Python compilation for the new file, running focused Mk5 tests, and running the existing Mk3/Mk4 outpainting regression tests.
