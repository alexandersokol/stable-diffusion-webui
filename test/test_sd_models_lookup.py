import importlib.util
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
        "modules.errors": types.SimpleNamespace(display=lambda *args, **kwargs: None),
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


def test_checkpoint_substring_lookup_uses_prebuilt_shortest_title_index():
    CheckpointLookupTests("test_checkpoint_substring_lookup_uses_prebuilt_shortest_title_index").test_checkpoint_substring_lookup_uses_prebuilt_shortest_title_index()


def test_checkpoint_registration_marks_title_index_for_lazy_rebuild():
    CheckpointLookupTests("test_checkpoint_registration_marks_title_index_for_lazy_rebuild").test_checkpoint_registration_marks_title_index_for_lazy_rebuild()


def test_checkpoint_info_uses_cached_legacy_model_hash():
    CheckpointLookupTests("test_checkpoint_info_uses_cached_legacy_model_hash").test_checkpoint_info_uses_cached_legacy_model_hash()


if __name__ == "__main__":
    unittest.main()
