import importlib.util
import sys
import types
import unittest
from pathlib import Path

import pytest


def load_util_for_test():
    util_path = Path(__file__).resolve().parents[1] / "modules" / "util.py"
    previous_shared = sys.modules.get("modules.shared")
    fake_shared = types.ModuleType("modules.shared")
    fake_shared.opts = types.SimpleNamespace(list_hidden_files=True)
    sys.modules["modules.shared"] = fake_shared

    try:
        spec = importlib.util.spec_from_file_location("_test_modules_util", util_path)
        util = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(util)
        return util, fake_shared
    finally:
        if previous_shared is None:
            sys.modules.pop("modules.shared", None)
        else:
            sys.modules["modules.shared"] = previous_shared


class WalkFilesTests(unittest.TestCase):
    def test_walk_files_yields_before_consuming_full_tree(self):
        util, _ = load_util_for_test()
        original_walk = util.os.walk
        calls = []

        def fake_walk(path, followlinks=True):
            calls.append(path)
            yield path, ["subdir"], ["root-2.txt", "root-1.ckpt", "root-1.txt"]
            calls.append(f"{path}/subdir")
            yield f"{path}/subdir", [], ["nested.txt"]

        try:
            util.os.walk = fake_walk
            walker = util.walk_files("models", allowed_extensions={".txt"})

            self.assertEqual(next(walker), util.os.path.join("models", "root-1.txt"))
            self.assertEqual(calls, ["models"])
            self.assertEqual([util.os.path.normpath(path) for path in walker], [
                util.os.path.join("models", "root-2.txt"),
                util.os.path.join("models", "subdir", "nested.txt"),
            ])
        finally:
            util.os.walk = original_walk


def test_walk_files_yields_before_consuming_full_tree():
    WalkFilesTests("test_walk_files_yields_before_consuming_full_tree").test_walk_files_yields_before_consuming_full_tree()


def test_walk_files_prunes_excluded_subtrees(tmp_path):
    util, _ = load_util_for_test()
    input_dir = tmp_path / "input"
    output_dir = input_dir / "output"
    source = input_dir / "source.png"
    prior_output = output_dir / "prior.png"
    output_dir.mkdir(parents=True)
    source.write_bytes(b"source")
    prior_output.write_bytes(b"prior output")

    discovered = list(util.walk_files(input_dir, excluded_paths=[output_dir]))

    assert [Path(path) for path in discovered] == [source]


def test_walk_files_prunes_symlink_to_excluded_subtree(tmp_path):
    util, _ = load_util_for_test()
    input_dir = tmp_path / "input"
    excluded_dir = tmp_path / "excluded"
    source = input_dir / "source.png"
    prior_output = excluded_dir / "prior.png"
    input_dir.mkdir()
    excluded_dir.mkdir()
    source.write_bytes(b"source")
    prior_output.write_bytes(b"prior output")

    try:
        (input_dir / "output-link").symlink_to(excluded_dir, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")

    discovered = list(util.walk_files(input_dir, excluded_paths=[excluded_dir]))

    assert [Path(path) for path in discovered] == [source]
