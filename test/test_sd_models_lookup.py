import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path


def load_sd_models_for_test(hashes_stub=None):
    import modules as modules_package

    sd_models_path = Path(__file__).resolve().parents[1] / "modules" / "sd_models.py"

    stubs = {
        "torch": types.SimpleNamespace(),
        "safetensors": types.ModuleType("safetensors"),
        "safetensors.torch": types.ModuleType("safetensors.torch"),
        "omegaconf": types.SimpleNamespace(OmegaConf=types.SimpleNamespace(), ListConfig=list),
        "ldm": types.ModuleType("ldm"),
        "ldm.modules": types.ModuleType("ldm.modules"),
        "ldm.modules.midas": types.ModuleType("ldm.modules.midas"),
        "tomesd": types.ModuleType("tomesd"),
        "modules.paths": types.SimpleNamespace(models_path="models"),
        "modules.shared": types.SimpleNamespace(
            opts=types.SimpleNamespace(),
            cmd_opts=types.SimpleNamespace(ckpt_dir=None),
            sd_model_file="",
            default_sd_model_file="",
            hf_endpoint="",
        ),
        "modules.modelloader": types.ModuleType("modules.modelloader"),
        "modules.devices": types.SimpleNamespace(cpu="cpu", device="cpu"),
        "modules.script_callbacks": types.ModuleType("modules.script_callbacks"),
        "modules.sd_vae": types.ModuleType("modules.sd_vae"),
        "modules.sd_disable_initialization": types.ModuleType("modules.sd_disable_initialization"),
        "modules.errors": types.SimpleNamespace(
            display=lambda *args, **kwargs: None,
            report=lambda *args, **kwargs: None,
        ),
        "modules.hashes": hashes_stub or types.SimpleNamespace(
            legacy_model_hash=lambda *args, **kwargs: "legacyhash",
            sha256_from_cache=lambda *args, **kwargs: None,
        ),
        "modules.sd_models_config": types.ModuleType("modules.sd_models_config"),
        "modules.sd_unet": types.ModuleType("modules.sd_unet"),
        "modules.sd_models_xl": types.ModuleType("modules.sd_models_xl"),
        "modules.cache": types.ModuleType("modules.cache"),
        "modules.extra_networks": types.ModuleType("modules.extra_networks"),
        "modules.processing": types.ModuleType("modules.processing"),
        "modules.lowvram": types.ModuleType("modules.lowvram"),
        "modules.sd_hijack": types.ModuleType("modules.sd_hijack"),
        "modules.patches": types.ModuleType("modules.patches"),
        "modules.model_cache_budget": types.ModuleType("modules.model_cache_budget"),
        "modules.timer": types.SimpleNamespace(Timer=object),
    }

    previous = {name: sys.modules.get(name) for name in stubs}
    previous_package_attrs = {}
    for name in stubs:
        if not name.startswith("modules."):
            continue

        attr_name = name.split(".", 1)[1]
        previous_package_attrs[attr_name] = (
            hasattr(modules_package, attr_name),
            getattr(modules_package, attr_name, None),
        )

    sys.modules.update(stubs)
    for name, module in stubs.items():
        if name.startswith("modules."):
            setattr(modules_package, name.split(".", 1)[1], module)

    try:
        spec = importlib.util.spec_from_file_location("_test_modules_sd_models", sd_models_path)
        sd_models = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sd_models)
        return sd_models
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

        for attr_name, (had_attr, value) in previous_package_attrs.items():
            if had_attr:
                setattr(modules_package, attr_name, value)
            elif hasattr(modules_package, attr_name):
                delattr(modules_package, attr_name)


class FakeCheckpointInfo:
    def __init__(self, title):
        self.title = title


def write_safetensors_metadata(filename, metadata):
    json_data = json.dumps({"__metadata__": metadata}).encode("utf8")
    with open(filename, "wb") as file:
        file.write(len(json_data).to_bytes(8, "little"))
        file.write(json_data)


class CheckpointLookupTests(unittest.TestCase):
    def test_checkpoint_info_uses_cached_legacy_model_hash(self):
        hash_calls = []
        hashes_stub = types.SimpleNamespace(
            legacy_model_hash=lambda filename, title: hash_calls.append((filename, title)) or "cachedhash",
            sha256_from_cache=lambda *args, **kwargs: None,
        )
        sd_models = load_sd_models_for_test(hashes_stub=hashes_stub)

        checkpoint_info = sd_models.CheckpointInfo("example.ckpt")

        self.assertEqual(checkpoint_info.hash, "cachedhash")
        self.assertEqual(hash_calls, [("example.ckpt", "checkpoint/example.ckpt")])

    def test_checkpoint_substring_lookup_uses_prebuilt_shortest_title_index(self):
        sd_models = load_sd_models_for_test()
        longer = FakeCheckpointInfo("z-folder/very-long-foo-model.safetensors [abc123]")
        shorter = FakeCheckpointInfo("a/foo.safetensors [def456]")

        sd_models.checkpoints_list.update({
            longer.title: longer,
            shorter.title: shorter,
        })
        sd_models.rebuild_checkpoint_title_search_index()

        def fail_if_lookup_sorts(*args, **kwargs):
            raise AssertionError("checkpoint lookup should not sort on every substring search")

        sd_models.sorted = fail_if_lookup_sorts

        self.assertIs(sd_models.get_closet_checkpoint_match("foo"), shorter)
        self.assertIs(sd_models.get_closet_checkpoint_match("foo [missing]"), shorter)

    def test_checkpoint_registration_marks_title_index_for_lazy_rebuild(self):
        sd_models = load_sd_models_for_test()
        checkpoint_info = sd_models.CheckpointInfo.__new__(sd_models.CheckpointInfo)
        checkpoint_info.title = "models/new-model.safetensors [abc123]"
        checkpoint_info.ids = []

        checkpoint_info.register()

        self.assertTrue(sd_models.checkpoint_title_search_index_dirty)
        self.assertIs(sd_models.get_closet_checkpoint_match("new-model"), checkpoint_info)
        self.assertFalse(sd_models.checkpoint_title_search_index_dirty)

    def test_safetensors_metadata_reader_skips_declared_headers_over_limit(self):
        sd_models = load_sd_models_for_test()
        sd_models.MAX_SAFETENSORS_METADATA_BYTES = 32
        read_sizes = []

        class FakeSafetensorsFile:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self, size=-1):
                read_sizes.append(size)
                if len(read_sizes) == 1:
                    return (sd_models.MAX_SAFETENSORS_METADATA_BYTES + 1).to_bytes(8, "little")
                if len(read_sizes) == 2:
                    return b'{"'
                raise AssertionError("oversized metadata body should not be read")

        sd_models.open = lambda *args, **kwargs: FakeSafetensorsFile()

        self.assertEqual(sd_models.read_metadata_from_safetensors("oversized.safetensors"), {})
        self.assertEqual(read_sizes, [8, 2])

    def test_safetensors_metadata_reader_keeps_oversized_nested_json_as_string(self):
        sd_models = load_sd_models_for_test()
        sd_models.MAX_SAFETENSORS_NESTED_METADATA_BYTES = 8
        metadata_file = Path(self._testMethodName + ".safetensors")
        nested_json = '{"large": true}'

        try:
            write_safetensors_metadata(metadata_file, {"sd_merge_recipe": nested_json})

            metadata = sd_models.read_metadata_from_safetensors(metadata_file)

            self.assertEqual(metadata["sd_merge_recipe"], nested_json)
        finally:
            metadata_file.unlink(missing_ok=True)

    def test_safetensors_metadata_reader_skips_oversized_thumbnail(self):
        sd_models = load_sd_models_for_test()
        sd_models.MAX_SAFETENSORS_THUMBNAIL_BYTES = 8
        metadata_file = Path(self._testMethodName + ".safetensors")

        try:
            write_safetensors_metadata(metadata_file, {
                "modelspec.thumbnail": "x" * 16,
                "author": "webui",
            })

            metadata = sd_models.read_metadata_from_safetensors(metadata_file)

            self.assertNotIn("modelspec.thumbnail", metadata)
            self.assertEqual(metadata["author"], "webui")
        finally:
            metadata_file.unlink(missing_ok=True)


def test_checkpoint_substring_lookup_uses_prebuilt_shortest_title_index():
    CheckpointLookupTests("test_checkpoint_substring_lookup_uses_prebuilt_shortest_title_index").test_checkpoint_substring_lookup_uses_prebuilt_shortest_title_index()


def test_checkpoint_registration_marks_title_index_for_lazy_rebuild():
    CheckpointLookupTests("test_checkpoint_registration_marks_title_index_for_lazy_rebuild").test_checkpoint_registration_marks_title_index_for_lazy_rebuild()


def test_checkpoint_info_uses_cached_legacy_model_hash():
    CheckpointLookupTests("test_checkpoint_info_uses_cached_legacy_model_hash").test_checkpoint_info_uses_cached_legacy_model_hash()


def test_safetensors_metadata_reader_skips_declared_headers_over_limit():
    CheckpointLookupTests("test_safetensors_metadata_reader_skips_declared_headers_over_limit").test_safetensors_metadata_reader_skips_declared_headers_over_limit()


def test_safetensors_metadata_reader_keeps_oversized_nested_json_as_string():
    CheckpointLookupTests("test_safetensors_metadata_reader_keeps_oversized_nested_json_as_string").test_safetensors_metadata_reader_keeps_oversized_nested_json_as_string()


def test_safetensors_metadata_reader_skips_oversized_thumbnail():
    CheckpointLookupTests("test_safetensors_metadata_reader_skips_oversized_thumbnail").test_safetensors_metadata_reader_skips_oversized_thumbnail()


if __name__ == "__main__":
    unittest.main()
