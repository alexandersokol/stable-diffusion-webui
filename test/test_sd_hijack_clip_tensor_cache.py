import sys
import types
import unittest
from unittest import mock

import torch


if "modules.sd_hijack" not in sys.modules:
    sd_hijack_stub = types.ModuleType("modules.sd_hijack")
    sd_hijack_stub.model_hijack = types.SimpleNamespace(
        embedding_db=types.SimpleNamespace(word_embeddings={}),
        fixes=None,
    )
    sys.modules["modules.sd_hijack"] = sd_hijack_stub

from modules import devices, sd_hijack_clip


class FakeTextConditionalModel(sd_hijack_clip.TextConditionalModel):
    def __init__(self):
        super().__init__()
        self.id_start = 101
        self.id_end = 102
        self.id_pad = 0
        self.encode_calls = 0
        self.token_tensor_ids = []
        self.multiplier_tensor_ids = []

    def tokenize(self, texts):
        raise NotImplementedError

    def encode_with_transformers(self, tokens):
        self.encode_calls += 1
        self.token_tensor_ids.append(id(tokens))
        return tokens.to(dtype=torch.float32).unsqueeze(-1)

    def encode_embedding_init_text(self, init_text, nvpt):
        raise NotImplementedError


class SdHijackClipTensorCacheTest(unittest.TestCase):
    def setUp(self):
        self.original_device = devices.device
        self.original_opts = sd_hijack_clip.opts

        devices.device = torch.device("cpu")
        sd_hijack_clip.opts = types.SimpleNamespace(
            emphasis="Ignore",
            CLIP_stop_at_last_layers=1,
            sdxl_clip_l_skip=False,
        )

    def tearDown(self):
        devices.device = self.original_device
        sd_hijack_clip.opts = self.original_opts

    def test_process_tokens_reuses_cached_token_and_multiplier_tensors_for_repeated_prompt_chunk(self):
        model = FakeTextConditionalModel()
        tokens = [[101, 11, 102, 102]]
        multipliers = [[1.0, 1.25, 1.0, 1.0]]
        original_asarray = torch.asarray

        class RecordingEmphasis:
            def after_transformers(emphasis_self):
                model.multiplier_tensor_ids.append(id(emphasis_self.multipliers))

        with (
            mock.patch("modules.sd_hijack_clip.sd_emphasis.get_current_option", return_value=RecordingEmphasis),
            mock.patch("modules.sd_hijack_clip.torch.asarray", side_effect=original_asarray) as asarray,
        ):
            first = model.process_tokens(tokens, multipliers)
            second = model.process_tokens(tokens, multipliers)

        self.assertEqual(2, model.encode_calls)
        self.assertEqual(2, asarray.call_count)
        self.assertEqual(model.token_tensor_ids[0], model.token_tensor_ids[1])
        self.assertEqual(model.multiplier_tensor_ids[0], model.multiplier_tensor_ids[1])
        self.assertTrue(torch.equal(first, second))

    def test_process_tokens_cache_misses_when_emphasis_changes(self):
        model = FakeTextConditionalModel()
        tokens = [[101, 11, 102, 102]]
        multipliers = [[1.0, 1.25, 1.0, 1.0]]
        original_asarray = torch.asarray

        class RecordingEmphasis:
            def after_transformers(emphasis_self):
                model.multiplier_tensor_ids.append(id(emphasis_self.multipliers))

        with (
            mock.patch("modules.sd_hijack_clip.sd_emphasis.get_current_option", return_value=RecordingEmphasis),
            mock.patch("modules.sd_hijack_clip.torch.asarray", side_effect=original_asarray) as asarray,
        ):
            model.process_tokens(tokens, multipliers)
            sd_hijack_clip.opts.emphasis = "None"
            model.process_tokens(tokens, multipliers)

        self.assertEqual(4, asarray.call_count)
        self.assertNotEqual(model.token_tensor_ids[0], model.token_tensor_ids[1])
        self.assertNotEqual(model.multiplier_tensor_ids[0], model.multiplier_tensor_ids[1])


if __name__ == "__main__":
    unittest.main()
