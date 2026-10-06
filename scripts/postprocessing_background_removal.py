import html
from pathlib import Path

import gradio as gr

from modules import background_removal, scripts_postprocessing, shared, ui_components


service = background_removal.BackgroundRemovalService()


class ScriptPostprocessingBackgroundRemoval(scripts_postprocessing.ScriptPostprocessing):
    name = "Background Removal"
    order = 5000
    terminal = True

    @staticmethod
    def _models_path() -> str:
        return shared.cmd_opts.background_removal_models_path

    def installed_models(self):
        return background_removal.discover_models(self._models_path())

    def model_choices(self):
        return list(self.installed_models())

    def ui(self):
        choices = self.model_choices()
        resolved_path = Path(self._models_path()).expanduser().resolve()
        escaped_path = html.escape(str(resolved_path))

        if choices:
            with ui_components.InputAccordion(False, label="Background Removal", elem_id="extras_background_removal") as enabled:
                model_name = gr.Dropdown(
                    label="Model",
                    choices=choices,
                    value=choices[0],
                    type="value",
                    elem_id="extras_background_removal_model",
                )
                gr.HTML(
                    f"<p>Models: <code>{escaped_path}</code>. "
                    "Outputs are saved as PNG to preserve transparency.</p>"
                )
        else:
            enabled = gr.Checkbox(value=False, visible=False, interactive=False)
            with gr.Accordion(label="Background Removal", open=False, elem_id="extras_background_removal"):
                model_name = gr.Dropdown(
                    label="Model",
                    choices=[],
                    value=None,
                    interactive=False,
                    elem_id="extras_background_removal_model",
                )
                gr.HTML(
                    f"<p>No supported models were found. Place registered ONNX files below "
                    f"<code>{escaped_path}</code> and restart the WebUI.</p>"
                )

        return {
            "enabled": enabled,
            "model_name": model_name,
        }

    def output_extension(self, default_extension, enabled, model_name):
        return "png" if enabled else default_extension

    def process(self, pp: scripts_postprocessing.PostprocessedImage, enabled, model_name):
        if not enabled:
            return

        if not model_name:
            raise background_removal.BackgroundRemovalError("A background-removal model must be selected")

        installed = self.installed_models()
        model = installed.get(model_name)
        if model is None:
            raise background_removal.BackgroundRemovalError(
                f"Background-removal model {model_name!r} is not available below {Path(self._models_path()).resolve()}"
            )

        pp.image = service.remove_background(pp.image, model)
        pp.info["Background removal model"] = model.spec.display_name
        pp.output_extension = "png"
