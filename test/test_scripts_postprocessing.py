from types import SimpleNamespace

from PIL import Image

from modules import shared, scripts_postprocessing


class NamedScript(scripts_postprocessing.ScriptPostprocessing):
    def __init__(self, name, order=1000, terminal=False):
        self.name = name
        self.order = order
        self.terminal = terminal


def runner_with(monkeypatch, scripts, preferred=()):
    monkeypatch.setattr(
        scripts_postprocessing.shared,
        "opts",
        SimpleNamespace(
            postprocessing_operation_order=list(preferred),
            postprocessing_disable_in_extras=[],
        ),
    )
    runner = scripts_postprocessing.ScriptPostprocessingRunner()
    runner.scripts = scripts
    return runner


def test_normal_scripts_keep_preference_then_order_sorting(monkeypatch):
    low_order = NamedScript("Low order", order=100)
    preferred = NamedScript("Preferred", order=900)
    runner = runner_with(monkeypatch, [low_order, preferred], preferred=["Preferred"])

    assert [script.name for script in runner.scripts_in_preferred_order()] == ["Preferred", "Low order"]


def test_terminal_script_stays_after_normal_scripts_despite_user_preference(monkeypatch):
    terminal = NamedScript("Background Removal", order=1, terminal=True)
    normal = NamedScript("Upscale", order=9999)
    runner = runner_with(monkeypatch, [terminal, normal], preferred=["Background Removal"])

    assert [script.name for script in runner.scripts_in_preferred_order()] == ["Upscale", "Background Removal"]


def test_terminal_scripts_keep_deterministic_preference_and_order(monkeypatch):
    alphabetical_second = NamedScript("Zulu", order=100, terminal=True)
    preferred = NamedScript("Preferred terminal", order=500, terminal=True)
    alphabetical_first = NamedScript("Alpha", order=100, terminal=True)
    runner = runner_with(
        monkeypatch,
        [alphabetical_second, preferred, alphabetical_first],
        preferred=["Preferred terminal"],
    )

    assert [script.name for script in runner.scripts_in_preferred_order()] == [
        "Preferred terminal",
        "Alpha",
        "Zulu",
    ]


def test_output_extension_hooks_run_in_final_script_order(monkeypatch):
    calls = []

    class FormatScript(NamedScript):
        def __init__(self, name, replacement, terminal=False):
            super().__init__(name, terminal=terminal)
            self.replacement = replacement
            self.controls = {"enabled": object()}

        def output_extension(self, default_extension, **args):
            calls.append((self.name, default_extension, args["enabled"]))
            return self.replacement if args["enabled"] else default_extension

    terminal = FormatScript("Terminal", "png", terminal=True)
    terminal.args_from, terminal.args_to = 0, 1
    normal = FormatScript("Normal", "webp")
    normal.args_from, normal.args_to = 1, 2
    runner = runner_with(monkeypatch, [terminal, normal], preferred=["Terminal"])

    result = runner.output_extension([True, True], "jpg")

    assert result == "png"
    assert calls == [("Normal", "jpg", True), ("Terminal", "webp", True)]


def test_default_output_extension_hook_keeps_requested_format():
    script = scripts_postprocessing.ScriptPostprocessing()

    assert script.output_extension("webp", enabled=True) == "webp"


def test_postprocessed_copy_preserves_preferred_output_extension():
    original = scripts_postprocessing.PostprocessedImage(Image.new("RGB", (1, 1)))
    original.output_extension = "png"

    copied = original.create_copy(Image.new("RGB", (2, 2)))

    assert copied.output_extension == "png"
