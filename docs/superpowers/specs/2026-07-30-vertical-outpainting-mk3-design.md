# Vertical Outpainting Mk3 Design

## Goal

Create a new img2img script named `Vertical Outpainting Mk3` that keeps the working behavior of `Poor man's outpainting`, focuses only on vertical expansion, and can automatically produce final vertical-placement variants from sequential outpainting passes.

## Scope

This is a new script, not a modification of `scripts/poor_mans_outpainting.py` or `scripts/outpainting_mk_2.py`.

The first version supports only vertical expansion. Horizontal directions, horizontal presets, zoom-out, and grids are out of scope.

## User Interface

The script appears only in img2img and is titled `Vertical Outpainting Mk3`.

Controls:

- `Pixels to expand`: slider from `8` to `384`, step `8`, default `384`.
- `Mask blur`: slider from `0` to `64`, step `1`, default `4`, matching `Poor man's outpainting`.
- `Masked content`: radio matching `Poor man's outpainting` choices and behavior.
- `Outpainting direction`: checkbox group with `up` and `down`, defaulting to both.
- `Shift preset`: dropdown with `All`, `Centered`, `Top`, `Bottom`, `Top Centered`, and `Bottom Centered`, defaulting to `All`.
- `Zoom in`: slider from `1.00` to `3.00`, step `0.05`, default `1.00`.
- `Inject continue prompt`: checkbox, default off.
- `Continue prompt`: textbox defaulting to `extend the image naturally, continue the existing scene, preserve composition, lighting, perspective, colors, style, and subject consistency`.

## Processing Model

Each selected preset starts from the same prepared input image.

If `Zoom in` is greater than `1.00`, the script scales the image content around the center, then center-crops it back to the original canvas dimensions. The original image file is not modified. From the script's perspective, the starting canvas size remains the same as the input image.

Outpainting is performed as sequential vertical passes. Each pass expands either up or down by its pass amount and then runs the same mask, latent mask, tiling, and `process_images` flow as `Poor man's outpainting`.

The script returns only completed final preset images. It does not return intermediate staged images from internal passes. It does not add or save a grid.

If `Inject continue prompt` is checked, the script appends the `Continue prompt` text to the prompt used for internal outpainting passes, then restores the original prompt after processing.

## Preset Semantics

All valid presets produce the same total vertical growth: `2 * Pixels to expand`.

For `Pixels to expand = P`, the available presets are:

- `Centered`: `P up`, then `P down`.
- `Top`: `P up`, then `P up`.
- `Bottom`: `P down`, then `P down`.
- `Top Centered`: `P up`, then `P / 2 down`, then `P / 2 up`.
- `Bottom Centered`: `P down`, then `P / 2 up`, then `P / 2 down`.

When `Shift preset` is `All`, the script runs all presets valid for the selected directions.

Direction filtering:

- If both `up` and `down` are selected, all presets are valid.
- If only `up` is selected, only `Top` is valid.
- If only `down` is selected, only `Bottom` is valid.
- If no direction is selected, the script returns `Processed(p, [], p.seed, "No vertical outpainting direction selected.")`.

Half-pass amounts use integer pixel values. Because the slider step is `8`, `P / 2` is always an integer.

## Seed And Batch Behavior

The script keeps the existing sequential seed progression style from `Poor man's outpainting`: after each internal `process_images` call, the next pass uses the returned seed plus one.

Each final preset variant is generated independently from the same prepared starting image. The first version follows `Poor man's outpainting` and processes one init image per script invocation by setting internal `n_iter` and `batch_size` to `1`. The output gallery contains final images only, so output count equals the number of selected valid presets.

## Saving And Metadata

If sample saving is enabled, each final preset image is saved as a sample. The script does not save a grid.

Generation metadata must include script-specific parameters in `p.extra_generation_params`, including the selected shift preset, selected directions, pixels to expand, zoom factor, and whether continue prompt injection was enabled.

## Testing

The implementation must include focused helper-level tests for:

- Building the preset pass sequences from direction and preset selections.
- Centered zoom preserving the original canvas size.
- Filtering invalid presets for single-direction selections.

Runtime verification must include importing the new script and running Python compilation for the new file.
