import importlib.util
import sys
import types
from pathlib import Path

import pytest


class FakeControl:
    def __init__(self, kind, value=None, **kwargs):
        self.kind = kind
        self.value = value
        self.kwargs = kwargs


class FakeGradio(types.ModuleType):
    def __init__(self):
        super().__init__("gradio")
        self.created = []

    def _control(self, kind, *args, **kwargs):
        value = args[0] if args else kwargs.pop("value", None)
        control = FakeControl(kind, value, **kwargs)
        self.created.append(control)
        return control

    def Markdown(self, *args, **kwargs):
        return self._control("Markdown", *args, **kwargs)

    def HTML(self, *args, **kwargs):
        return self._control("HTML", *args, **kwargs)

    def Slider(self, *args, **kwargs):
        return self._control("Slider", *args, **kwargs)

    def Radio(self, *args, **kwargs):
        return self._control("Radio", *args, **kwargs)

    def CheckboxGroup(self, *args, **kwargs):
        return self._control("CheckboxGroup", *args, **kwargs)

    def Dropdown(self, *args, **kwargs):
        return self._control("Dropdown", *args, **kwargs)

    def Checkbox(self, *args, **kwargs):
        return self._control("Checkbox", *args, **kwargs)

    def Textbox(self, *args, **kwargs):
        return self._control("Textbox", *args, **kwargs)


class FakeScriptBase:
    def elem_id(self, item_id):
        return item_id


def load_script_module(script_name):
    import modules as modules_package

    script_path = Path(__file__).resolve().parents[1] / "scripts" / script_name
    fake_gradio = FakeGradio()
    stubs = {
        "gradio": fake_gradio,
        "modules.scripts": types.SimpleNamespace(Script=FakeScriptBase),
        "modules.images": types.ModuleType("modules.images"),
        "modules.devices": types.SimpleNamespace(torch_gc=lambda: None),
        "modules.processing": types.SimpleNamespace(
            Processed=object,
            process_images=lambda *args, **kwargs: None,
            create_infotext=lambda *args, **kwargs: "",
        ),
        "modules.shared": types.SimpleNamespace(opts=types.SimpleNamespace(), state=types.SimpleNamespace()),
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
        spec = importlib.util.spec_from_file_location(f"_test_{Path(script_name).stem}", script_path)
        script_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script_module)
        return script_module, fake_gradio
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


@pytest.mark.parametrize(
    ("script_name", "expected_title", "expected_text"),
    [
        ("outpainting_mk_2.py", "Outpainting mk2", "recommended settings"),
        ("poor_mans_outpainting.py", "Poor man's outpainting", "masked generation"),
        ("vertical_outpainting_mk3.py", "Vertical Outpainting Mk3", "shift presets"),
        ("vertical_outpainting_mk4.py", "Vertical Outpainting Mk4", "target height"),
        ("vertical_outpainting_mk5.py", "Vertical Outpainting Mk5", "scale"),
    ],
)
def test_outpainting_scripts_show_top_description_without_changing_run_args(script_name, expected_title, expected_text):
    script_module, fake_gradio = load_script_module(script_name)

    controls = script_module.Script().ui(True)

    markdowns = [control for control in fake_gradio.created if control.kind == "Markdown"]
    assert len(markdowns) == 1
    assert fake_gradio.created[0] is markdowns[0]
    assert expected_title in markdowns[0].value
    assert expected_text in markdowns[0].value.lower()
    assert markdowns[0] not in controls
