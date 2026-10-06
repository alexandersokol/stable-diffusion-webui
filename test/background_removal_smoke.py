"""Opt-in real-model smoke test for the Extras background-removal subsystem."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from modules import background_removal  # noqa: E402
from modules.paths_internal import models_path  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-path",
        type=Path,
        default=Path(models_path) / "bg-removal",
        help="Directory containing registered background-removal ONNX files.",
    )
    parser.add_argument("--image", type=Path, required=True, help="Reference image used for inference.")
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Display name to run; repeat for multiple models. The default runs every installed model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for PNG outputs. No images are saved when omitted.",
    )
    return parser


def select_models(installed, requested):
    if not requested:
        return list(installed.values())

    missing = [name for name in requested if name not in installed]
    if missing:
        available = ", ".join(installed) or "none"
        raise ValueError(f"Unavailable model(s): {', '.join(missing)}. Installed models: {available}")

    return [installed[name] for name in requested]


def validate_result(source, result, model_name):
    if result.mode != "RGBA":
        raise ValueError(f"{model_name} returned mode {result.mode}, expected RGBA")
    if result.size != source.size:
        raise ValueError(f"{model_name} returned size {result.size}, expected {source.size}")

    alpha = np.asarray(result.getchannel("A"), dtype=np.float32)
    if not np.isfinite(alpha).all():
        raise ValueError(f"{model_name} returned a non-finite alpha channel")

    alpha_min = float(alpha.min())
    alpha_max = float(alpha.max())
    if alpha_min == alpha_max:
        raise ValueError(f"{model_name} returned a constant alpha channel ({alpha_min:g})")

    return alpha_min, alpha_max


def run(args):
    installed = background_removal.discover_models(args.models_path)
    if not installed:
        print(f"No registered models found below {args.models_path.resolve()}", file=sys.stderr)
        return 2

    try:
        selected = select_models(installed, args.models)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    with Image.open(args.image) as opened:
        source = opened.copy()

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    service = background_removal.BackgroundRemovalService()
    failures = 0
    try:
        for model in selected:
            started = time.perf_counter()
            try:
                result = service.remove_background(source, model)
                alpha_min, alpha_max = validate_result(source, result, model.spec.display_name)
                elapsed = time.perf_counter() - started
                providers = ", ".join(service.active_providers) or "unknown"
                print(
                    f"PASS {model.spec.display_name}: provider={providers}; elapsed={elapsed:.3f}s; "
                    f"size={result.width}x{result.height}; mode={result.mode}; "
                    f"alpha={alpha_min:g}..{alpha_max:g}"
                )

                if args.output_dir is not None:
                    output_path = args.output_dir / f"{model.path.stem}.png"
                    result.save(output_path, format="PNG")
                    print(f"  saved {output_path.resolve()}")
            except Exception as exc:
                failures += 1
                print(f"FAIL {model.spec.display_name}: {exc}", file=sys.stderr)
    finally:
        service.unload()

    return 1 if failures else 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
