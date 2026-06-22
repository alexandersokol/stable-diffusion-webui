from types import SimpleNamespace
import sys
import types

import torch

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


def test_decode_latent_batch_uses_configured_chunks(monkeypatch):
    calls = []

    def fake_decode_first_stage(model, batch):
        calls.append(batch.shape[0])
        return batch

    monkeypatch.setattr(processing, "decode_first_stage", fake_decode_first_stage)
    monkeypatch.setattr(processing.shared, "opts", SimpleNamespace(final_vae_decode_chunk_size=2, auto_vae_precision=False, auto_vae_precision_bfloat16=False))
    monkeypatch.setattr(processing.devices, "test_for_nans", lambda *_args, **_kwargs: None)

    batch = torch.zeros((5, 3, 2, 2))

    samples = processing.decode_latent_batch(SimpleNamespace(first_stage_model=SimpleNamespace(to=lambda *_args, **_kwargs: None)), batch)

    assert calls == [2, 2, 1]
    assert len(samples) == 5
