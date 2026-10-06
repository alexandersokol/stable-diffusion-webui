import os

from modules import cmd_args, shared
from modules.paths_internal import models_path, normalized_filepath


def test_background_removal_models_path_defaults_below_models_directory():
    expected = os.path.join(models_path, "bg-removal")

    assert shared.cmd_opts.background_removal_models_path == expected


def test_background_removal_models_path_normalizes_an_override(tmp_path):
    requested = tmp_path / "models" / ".." / "external-bg-models"

    parsed, _unknown = cmd_args.parser.parse_known_args([
        "--background-removal-models-path",
        str(requested),
    ])

    assert parsed.background_removal_models_path == normalized_filepath(str(requested))
