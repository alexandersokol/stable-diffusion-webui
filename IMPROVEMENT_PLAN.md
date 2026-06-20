# Improvement Plan

This document tracks resource, performance, and reliability improvements for this Stable Diffusion WebUI checkout. Scope: core project files plus `extensions-builtin`; custom extensions under `extensions/` are intentionally excluded. Code style-only cleanup is out of scope.

## Critical

| Id | name | description | status |
| --- | --- | --- | --- |
| 1 | Image filename collision overwrite | `modules/images.py` searches only 500 numbered filenames before saving; if all candidates already exist, the last candidate can still be used and replaced. Add an explicit failure or continue searching safely to prevent accidental image loss. | done |
| 2 | Model merger OOM and partial output risk | `modules/extras.py` can load two or three full checkpoints plus optional VAE tensors into RAM and saves directly to the target model path. Add memory/disk preflight checks and atomic output writing so failed merges do not leave corrupt checkpoints. | done |
| 3 | Unsafe temp cleanup policy | `modules/ui_tempdir.py` deletes every `.png` under a configured temp directory at startup and ignores non-PNG temp files. Track WebUI-owned files and use TTL/ownership-based cleanup to avoid both storage leaks and accidental deletion of unrelated PNGs. | done |
| 4 | Progress state lacks synchronization | `modules/progress.py` stores `current_task`, queues, finished tasks, and recorded results in module globals without locks. Add thread-safe access because UI/API polling and queued generation can run concurrently and leave stale or inconsistent status. | done |

## High

| Id | name | description | status |
| --- | --- | --- | --- |
| 5 | Live preview resource pressure | `modules/progress.py`, `modules/shared_state.py`, and `javascript/progressbar.js` repeatedly decode, grid, encode, base64, and transmit previews. Add size caps, adaptive refresh, browser visibility throttling, and server-side preview reuse. | done |
| 6 | Current preview retained after job end | `State.end()` does not clear `current_latent` or `current_image`, so large preview images can remain in RAM after generation until the next job begins. Clear preview tensors/images when no restore path needs them. | done |
| 7 | Large batches retained in memory | `modules/processing.py` keeps all output PIL images, optional grids, masks, and infotexts in memory until the response is returned. Add streaming/limited gallery return modes or spill large batches to temp files. | done |
| 8 | Upscale cache holds large images | `scripts/postprocessing_upscale.py` hashes full image pixels and stores full PIL results in `upscale_cache` with only an item-count limit. Replace with byte-budgeted LRU cache and cheaper cache keys. | done |
| 9 | Checkpoint RAM cache has no memory budget | `modules/sd_models.py` can keep multiple checkpoints in CPU RAM when `sd_checkpoints_limit` is increased. Add estimated memory accounting, UI warnings, and eviction by memory pressure rather than count only. | backlog |
| 10 | Built-in upscaler model retention | `extensions-builtin/SwinIR` caches the loaded model on device, while `extensions-builtin/LDSR` exposes a cache option but reload behavior is not clearly bounded. Add explicit unload/cache controls and memory reporting for built-in upscalers. | backlog |
| 11 | Recorded progress results retain images | `modules/progress.py` keeps recent `Processed` results for restore. Even with a small count, each result can contain large images. Store lightweight references or temp-file-backed results instead of full image objects. | backlog |
| 12 | LoRA weight backups can grow CPU RAM | `extensions-builtin/Lora/networks.py` stores CPU backup weights on patched layers. Audit lifecycle and add cleanup/telemetry so frequent LoRA changes do not retain unnecessary backup tensors. | backlog |

## Medium

| Id | name | description | status |
| --- | --- | --- | --- |
| 13 | Output directory scans are O(n) | `modules/images.py:get_next_sequence_number()` scans the whole output directory for every save. Cache sequence counters per directory/basename and resync safely when files appear externally. | backlog |
| 14 | Disk cache aggregate can grow large | `modules/cache.py` sets a 4 GB limit per subsection, so total cache size can exceed expectations. Add global cache budget, cleanup UI, and periodic culling across subsections. | backlog |
| 15 | Extra Networks HTML generation scales poorly | `modules/ui_extra_networks.py` loads metadata and creates card HTML for all items at once. Add pagination or lazy card rendering for large checkpoint/embedding libraries. | backlog |
| 16 | Interrogate text ranking recomputes too much | `modules/interrogate.py` tokenizes and encodes category text repeatedly. Cache tokenized text and category text features by model/device/options to reduce CPU/GPU work. | backlog |
| 17 | Tiled upscaling allocates full tensors on GPU | `modules/upscaler_utils.py:tiled_upscale_2()` creates full-size result and weight tensors on the target device. Add a CPU accumulation mode or chunked output path for very large upscale jobs. | backlog |
| 18 | Frequent cache clearing can stall generation | `devices.torch_gc()` is called at many generation boundaries and can force device synchronization. Profile call sites and make GC adaptive based on memory pressure and backend. | backlog |
| 19 | Model metadata reads can be expensive | Checkpoint, LoRA, and Extra Networks metadata paths read large files or safetensors metadata during refresh. Add async refresh progress, cancellation, and stronger cached metadata invalidation. | backlog |
| 20 | UI progress polling has no global coordinator | Multiple pages or restored sessions can start independent progress loops. Centralize progress polling per task and fan out updates to views to reduce duplicate network requests. | backlog |

## Low

| Id | name | description | status |
| --- | --- | --- | --- |
| 21 | Storage-heavy options need guardrails | Options such as grids, masks, before/after correction images, init image saving, and text sidecars can multiply output storage. Add estimated per-job storage warnings for large batches. | backlog |
| 22 | Profiling output can be very large | Torch profiler settings can create hundreds of MB per run. Add rotation, explicit size warnings, and a cleanup button for profiling traces. | backlog |
| 23 | Temp theme/cache directories lack UI cleanup | `tmp/gradio_themes`, moved corrupt config/cache files, and other auxiliary files can accumulate. Add a maintenance action that reports and cleans known safe temporary locations. | backlog |
| 24 | Console progress overhead is always user-managed | `multiple_tqdm` and tiled upscale progress bars can add overhead in long runs. Add auto-disable heuristics for non-interactive or high-throughput sessions. | backlog |
| 25 | Image save sidecar logs can grow forever | `log.csv` and optional `.txt` infotext sidecars have no retention or size visibility. Add log rotation and output folder usage summaries. | backlog |
| 26 | API image responses duplicate memory | API encode paths convert generated PIL images into in-memory base64 responses. Add optional file-reference responses or streaming for large batches. | backlog |
