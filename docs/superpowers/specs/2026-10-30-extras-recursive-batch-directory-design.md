# Recursive Extras Batch Directory Design

## Intent

Make Extras → Batch from Directory process images recursively and preserve each source image's relative directory beneath the selected output root. This must apply to the entire Extras postprocessing chain, including background removal, without changing Single Image or Batch Process behavior.

## Goals

- Discover files in the input directory and all of its subdirectories using deterministic natural ordering.
- Save every processed image beneath the output root at the same relative parent path it had beneath the input root.
- Use the configured default Extras output directory as the output root when the UI Output directory field is blank.
- Keep output-format selection, original-name behavior, extra-image suffixes, captions, gallery results, progress counts, and interruption behavior unchanged.
- Make "Skip existing files" check the mirrored destination directory for each source image and each effective output extension.
- Prevent an output directory located strictly beneath the input directory from being discovered as input.

## Non-goals

- Do not change recursive behavior for img2img, interrogation, API batch uploads, or Extras → Batch Process.
- Do not copy empty input directories; create only directories needed for saved outputs.
- Do not copy source metadata or non-image files beyond the behavior already provided by the Extras save pipeline.
- Do not add a recursion toggle: recursive traversal is the default behavior for Batch from Directory.

## Constraints and Edge Cases

- Preserve the current file-reading behavior: traversal may encounter any file, while unreadable/non-image files are skipped by the existing image loader.
- Reuse `modules.util.walk_files()` so hidden-directory policy, natural sorting, symlink traversal, and cycle prevention stay consistent with the rest of the project.
- Extend `walk_files()` with an optional excluded-path collection. Existing callers that omit it retain their current behavior.
- Compare excluded directories using normalized real paths so a symlink into an excluded output tree is also pruned.
- Exclude the output root only when it is a strict descendant of the input root. If input and output are the same directory, process sources normally and save results beside them in their corresponding directories.
- Derive destination parents from the source path relative to the input root. A root-level source saves directly under the output root; `input/people/a.png` saves under `output/people/`.
- Files with identical basenames in different input subdirectories remain independent because their destination directories differ.
- A nested existing output must be pruned before the input manifest and job count are built, so prior outputs are never treated as new inputs.

## Selected Approach

Extend `modules.util.walk_files(path, allowed_extensions=None, excluded_paths=None)` with optional subtree pruning, then use it from Extras directory mode. `run_postprocessing()` will carry a relative parent directory with each directory-mode work item and compute an item-specific destination path for preflight existence checks, image saves, extra images, and captions.

This approach was selected over an Extras-only walker because it avoids duplicating existing hidden-file, ordering, symlink, and cycle logic. It was selected over filtering yielded files afterward because pruning avoids traversing a potentially large nested output tree.

## Alternatives Considered

### Extras-only recursive walker

This keeps the utility API untouched but duplicates traversal and safety behavior in `modules/postprocessing.py`, creating two implementations that could drift.

### Filter output files after traversal

This is the smallest change, but still walks the entire output subtree and builds unnecessary paths before rejecting them.

## Code Boundaries

- `modules/util.py`: add backward-compatible excluded-subtree support to `walk_files()`.
- `modules/postprocessing.py`: recursively build directory-mode work items, derive mirrored output parents, and use those parents for skip/save behavior.
- `modules/ui_postprocessing.py`: clarify in the Batch from Directory UI that subdirectories are processed and mirrored.
- `test/test_walk_files.py`: cover excluded-directory pruning, including normalized/symlink paths where supported.
- `test/test_extras.py`: cover recursive discovery, mirrored paths, duplicate basenames, default output roots, nested-output exclusion, and nested skip-existing behavior.

## Verification

- Run the focused utility and Extras unit tests.
- Run the existing background-removal and postprocessing tests to confirm the full Extras chain still works.
- Run `git diff --check` and confirm no unrelated files changed.
