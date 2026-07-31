# Vertical Outpainting Soft Seam Blend Design

## Goal

Reduce the visible line that can appear where generated outpaint pixels meet the original or prepared source pixels in `Vertical Outpainting Mk3`, `Vertical Outpainting Mk4`, and `Vertical Outpainting Mk5`.

The fix should be deterministic, should not add another generation pass, and should keep the existing scripts' core behavior intact.

## Problem

The current scripts can generate a plausible continuation, but the final image may still show a thin seam. The likely cause is the final restoration step: the scripts protect the source by pasting original or prepared source pixels back over the generated image. That exact paste is useful for preserving the source, but it creates a hard boundary between generated pixels and source pixels.

Mask blur helps the inpainting pass, but it does not soften the final paste edge. The seam can therefore remain visible even when the generated content is otherwise correct.

## Scope

This change applies to:

- `scripts/vertical_outpainting_mk3.py`
- `scripts/vertical_outpainting_mk4.py`
- `scripts/vertical_outpainting_mk5.py`

This change does not modify `poor_mans_outpainting.py` or `outpainting_mk_2.py`.

This change does not add a second inpainting pass, new sampler controls, ControlNet/IP-Adapter wiring, or any new output variants.

## User Interface

Each vertical outpainting script gets a new control:

- Label: `Soft seam blend`
- Type: slider
- Range: `0` to `128`
- Step: `8`
- Default: `32`

`0` disables the feature and preserves current hard-paste behavior.

The control should appear near `Mask blur` and `Seam size` where those controls exist. Mk3 does not currently expose `Seam size`, so the new control should appear near `Mask blur`.

## Behavior

Soft seam blend is a post-processing operation applied after generation and before final save/return.

For each vertical edge where generated pixels meet source pixels:

- Build an alpha ramp over the selected blend width.
- Keep generated pixels dominant on the generated side of the boundary.
- Keep source pixels dominant deeper inside the source.
- Leave the protected source interior exact outside the blend band.
- Leave generated gap pixels untouched outside the blend band.

If `Soft seam blend = 0`, the scripts must produce the same final image as they do today.

The blend width is clamped so it never exceeds the available source area or final image bounds.

## Mk3 Details

Mk3 expands through sequential passes. The blend should be applied at the boundary for each completed pass, after each pass has produced its result and after the source/current image region has been restored for that pass.

This is preferred over blending only the very final result because intermediate passes become the input to later passes. If a hard seam is carried forward into the next pass, later generation can reinforce it.

The script must still return only completed preset outputs, not staged internal images.

## Mk4 Details

Mk4 already has `Source handling` options:

- `Preserve source pixels`
- `Blend seam only`
- `Soft resynthesis`

Soft seam blend applies to the modes that restore or protect source pixels:

- In `Preserve source pixels`, blend only the boundary band and keep the source interior exact.
- In `Blend seam only`, use the same final feathering behavior around the seam band.
- In `Soft resynthesis`, do not force an extra source paste; the generation already owns the seam and source region according to that mode.

## Mk5 Details

Mk5 restores the prepared scaled/cropped source exactly after generation. Soft seam blend should replace the hard transition with a feathered transition at the top and/or bottom prepared-source boundary when that boundary touches generated space.

For `Scale = 1.00` or any no-mask/source-only path, there is no generated seam. The feature should not alter the source-only result.

The final image width must remain the source width. Final metadata and save behavior must continue to use the final dimensions.

## Metadata

Each script should record the selected value in `p.extra_generation_params`:

- Mk3: `Vertical Outpainting MK3 soft seam blend`
- Mk4: `Vertical Outpainting MK4 soft seam blend`
- Mk5: `Vertical Outpainting MK5 soft seam blend`

Existing metadata should remain unchanged.

## Testing

Tests should cover:

- `Soft seam blend = 0` preserves the existing hard-paste result.
- A nonzero blend changes only the seam band.
- The protected source interior remains exact outside the blend band.
- The generated gap area remains unchanged outside the blend band.
- Mk5 source-only / scale-one path is not altered.
- Mk3, Mk4, and Mk5 still pass their existing outpainting regression tests.

Verification should include:

- Focused tests for the new blend helper behavior.
- Existing Mk3, Mk4, and Mk5 outpainting tests.
- Python compilation for all three edited scripts and updated tests.

## Recommended Defaults

Defaulting `Soft seam blend` to `32` gives a visible seam reduction for common `1024x1280 -> 1024x2048` workflows while keeping most of the source untouched.

Users who need exact source preservation can set the slider to `0`.
