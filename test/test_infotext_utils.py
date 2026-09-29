import importlib
import sys
import types
import unittest


MISSING = object()
STUBBED_MODULES = (
    "gradio",
    "PIL",
    "PIL.Image",
    "modules.paths",
    "modules.shared",
    "modules.ui_tempdir",
    "modules.script_callbacks",
    "modules.processing",
    "modules.infotext_versions",
    "modules.images",
    "modules.prompt_parser",
    "modules.errors",
    "modules.infotext_utils",
    "modules.generation_parameters_copypaste",
)


def import_infotext_utils():
    original_modules = {name: sys.modules.get(name, MISSING) for name in STUBBED_MODULES}
    update_type = type("Update", (), {})

    gradio_stub = types.SimpleNamespace(
        update=lambda **kwargs: update_type(),
        components=types.SimpleNamespace(Component=object),
        Gallery=type("Gallery", (), {}),
        Dropdown=types.SimpleNamespace(update=lambda **kwargs: {"__type__": "generic_update", **kwargs}),
    )

    shared_opts = types.SimpleNamespace(
        infotext_skip_pasting=[],
        infotext_styles="Ignore",
        use_old_hires_fix_width_height=False,
    )
    shared_stub = types.SimpleNamespace(opts=shared_opts)

    stubs = {
        "gradio": gradio_stub,
        "PIL": types.ModuleType("PIL"),
        "PIL.Image": types.SimpleNamespace(Image=object),
        "modules.paths": types.SimpleNamespace(data_path="."),
        "modules.shared": shared_stub,
        "modules.ui_tempdir": types.SimpleNamespace(check_tmp_file=lambda demo, filename: True),
        "modules.script_callbacks": types.SimpleNamespace(infotext_pasted_callback=lambda prompt, params: None),
        "modules.processing": types.SimpleNamespace(old_hires_fix_first_pass_dimensions=lambda width, height: (width, height)),
        "modules.infotext_versions": types.SimpleNamespace(backcompat=lambda params: None, parse_version=lambda version: None),
        "modules.images": types.SimpleNamespace(read=lambda source: None),
        "modules.prompt_parser": types.SimpleNamespace(parse_prompt_attention=lambda prompt: []),
        "modules.errors": types.SimpleNamespace(report=lambda *args, **kwargs: None),
    }

    for name, stub in stubs.items():
        sys.modules[name] = stub

    sys.modules.pop("modules.infotext_utils", None)
    return importlib.import_module("modules.infotext_utils"), original_modules


def restore_modules(original_modules):
    for name, module in original_modules.items():
        if module is MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


class ParseGenerationParametersTest(unittest.TestCase):
    def load_infotext_utils(self):
        infotext_utils, original_modules = import_infotext_utils()
        self.addCleanup(restore_modules, original_modules)
        return infotext_utils

    def test_flattened_negative_prompt_on_parameter_line_is_preserved(self):
        infotext_utils = self.load_infotext_utils()

        infotext = (
            "a cat\n"
            "Negative prompt: blurry, low quality, Steps: 20, Sampler: Euler a, "
            "CFG scale: 7, Seed: 123, Size: 512x512"
        )

        params = infotext_utils.parse_generation_parameters(infotext, [])

        self.assertEqual("blurry, low quality", params["Negative prompt"])
        self.assertEqual("20", params["Steps"])

    def test_single_line_infotext_with_negative_prompt_is_preserved(self):
        infotext_utils = self.load_infotext_utils()

        infotext = (
            "a cat Negative prompt: blurry, low quality, Steps: 20, Sampler: Euler a, "
            "CFG scale: 7, Seed: 123, Size: 512x512"
        )

        params = infotext_utils.parse_generation_parameters(infotext, [])

        self.assertEqual("a cat", params["Prompt"])
        self.assertEqual("blurry, low quality", params["Negative prompt"])
        self.assertEqual("20", params["Steps"])


if __name__ == "__main__":
    unittest.main()
