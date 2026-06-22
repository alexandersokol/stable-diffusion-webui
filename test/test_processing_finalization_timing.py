from types import SimpleNamespace
import sys
import types

sd_hijack_stub = types.ModuleType("modules.sd_hijack")
sd_hijack_stub.model_hijack = SimpleNamespace(extra_generation_params={}, comments=[])
sys.modules.setdefault("modules.sd_hijack", sd_hijack_stub)
sys.modules.setdefault("modules.prompt_parser", types.ModuleType("modules.prompt_parser"))
sys.modules.setdefault("modules.masking", types.ModuleType("modules.masking"))
sys.modules.setdefault("modules.sd_samplers", types.ModuleType("modules.sd_samplers"))
sys.modules.setdefault("modules.lowvram", SimpleNamespace(is_enabled=lambda *_args, **_kwargs: False, send_everything_to_cpu=lambda: None))
sys.modules.setdefault("modules.infotext_utils", types.ModuleType("modules.infotext_utils"))
sys.modules.setdefault("modules.extra_networks", SimpleNamespace(deactivate=lambda *_args, **_kwargs: None))
sys.modules.setdefault("modules.sd_vae_approx", types.ModuleType("modules.sd_vae_approx"))
sys.modules.setdefault("modules.scripts", SimpleNamespace(ScriptRunner=object))
sd_samplers_common_stub = types.ModuleType("modules.sd_samplers_common")
sd_samplers_common_stub.images_tensor_to_samples = lambda *args, **kwargs: None
sd_samplers_common_stub.decode_first_stage = lambda _model, batch: batch
sd_samplers_common_stub.approximation_indexes = {}
sys.modules.setdefault("modules.sd_samplers_common", sd_samplers_common_stub)
sys.modules.setdefault("modules.sd_unet", types.ModuleType("modules.sd_unet"))
sys.modules.setdefault("modules.rng", SimpleNamespace(slerp=lambda *args, **kwargs: None, ImageRNG=object))
sys.modules.setdefault("modules.face_restoration", types.ModuleType("modules.face_restoration"))
sys.modules.setdefault("modules.sd_models", types.ModuleType("modules.sd_models"))
sys.modules.setdefault("modules.sd_vae", types.ModuleType("modules.sd_vae"))
sys.modules.setdefault("modules.styles", types.ModuleType("modules.styles"))
sys.modules.setdefault("modules.sd_schedulers", types.ModuleType("modules.sd_schedulers"))

ldm_data_util_stub = types.ModuleType("ldm.data.util")
ldm_data_util_stub.AddMiDaS = object
sys.modules.setdefault("ldm", types.ModuleType("ldm"))
sys.modules.setdefault("ldm.data", types.ModuleType("ldm.data"))
sys.modules.setdefault("ldm.data.util", ldm_data_util_stub)
ldm_ddpm_stub = types.ModuleType("ldm.models.diffusion.ddpm")
ldm_ddpm_stub.LatentDepth2ImageDiffusion = object
sys.modules.setdefault("ldm.models", types.ModuleType("ldm.models"))
sys.modules.setdefault("ldm.models.diffusion", types.ModuleType("ldm.models.diffusion"))
sys.modules.setdefault("ldm.models.diffusion.ddpm", ldm_ddpm_stub)

from modules import processing


def test_add_finalization_timing_comment_records_nonzero_phases():
    p = SimpleNamespace(comments=[])
    timings = {"VAE decode": 0.25, "Image save": 0.5, "Tensor to image": 0.0}

    processing.add_finalization_timing_comment(p, timings)

    assert p.comments == ["Finalization: VAE decode 0.25s, Image save 0.50s"]
