import sys
import tempfile
from types import SimpleNamespace

import torch

from modules import interrogate


class FakeClipModule:
    def __init__(self):
        self.tokenize_calls = 0

    def tokenize(self, texts, truncate=True):
        self.tokenize_calls += 1
        rows = []
        for index, _text in enumerate(texts, start=1):
            rows.append([float(index), float(index + 1)])

        return torch.tensor(rows, dtype=torch.float32)


class FakeClipModel:
    def __init__(self):
        self.encode_text_calls = 0

    def encode_text(self, tokens):
        self.encode_text_calls += 1
        return tokens


def create_interrogator(dict_limit=0):
    interrogate.shared.opts = SimpleNamespace(
        interrogate_clip_dict_limit=dict_limit,
        interrogate_clip_skip_categories=[],
    )
    interrogate.devices.device_interrogate = torch.device("cpu")
    interrogate.devices.torch_gc = lambda: None

    model = interrogate.InterrogateModels("unused")
    model.clip_model = FakeClipModel()
    model.dtype = torch.float32
    model.refresh_text_feature_cache_model_key()

    fake_clip = FakeClipModule()
    sys.modules["clip"] = fake_clip

    return model, fake_clip


def test_rank_reuses_cached_text_features_for_same_category():
    model, fake_clip = create_interrogator()
    image_features = torch.tensor([[1.0, 1.0]])
    text_items = ["first", "second", "third"]

    model.rank(image_features, text_items, category_name="artists")
    model.rank(image_features, text_items, category_name="artists")

    assert fake_clip.tokenize_calls == 1
    assert model.clip_model.encode_text_calls == 1
    assert len(model.text_feature_cache) == 1


def test_rank_batches_similarity_for_multiple_image_features(monkeypatch):
    model, _fake_clip = create_interrogator()
    image_features = torch.tensor([[1.0, 0.5], [0.25, 1.5]], dtype=torch.float32)
    text_items = ["first", "second", "third"]
    _limited_text_array, text_features = model.cached_text_features(text_items, "artists")
    expected_similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1).mean(dim=0, keepdim=True)
    expected_probs, expected_labels = expected_similarity.cpu().topk(2, dim=-1)
    expected = [
        (text_items[expected_labels[0][i].numpy()], expected_probs[0][i].numpy() * 100)
        for i in range(2)
    ]
    softmax_shapes = []
    original_softmax = torch.Tensor.softmax

    def recording_softmax(self, *args, **kwargs):
        softmax_shapes.append(tuple(self.shape))
        return original_softmax(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "softmax", recording_softmax)

    actual = model.rank(image_features, text_items, top_count=2, category_name="artists")

    assert actual == expected
    assert softmax_shapes == [(2, 3)]


def test_rank_recomputes_text_features_when_dict_limit_changes():
    model, fake_clip = create_interrogator(dict_limit=2)
    image_features = torch.tensor([[1.0, 1.0]])
    text_items = ["first", "second", "third"]

    model.rank(image_features, text_items, category_name="artists")
    interrogate.shared.opts.interrogate_clip_dict_limit = 3
    model.rank(image_features, text_items, category_name="artists")

    assert fake_clip.tokenize_calls == 2
    assert model.clip_model.encode_text_calls == 2
    assert len(model.text_feature_cache) == 1


def test_refresh_model_key_clears_old_text_feature_cache():
    model, _fake_clip = create_interrogator()
    image_features = torch.tensor([[1.0, 1.0]])
    text_items = ["first", "second", "third"]

    model.rank(image_features, text_items, category_name="artists")
    assert len(model.text_feature_cache) == 1

    old_clip_model = model.clip_model
    model.clip_model = FakeClipModel()
    model.refresh_text_feature_cache_model_key()
    assert len(model.text_feature_cache) == 0

    model.rank(image_features, text_items, category_name="artists")

    assert old_clip_model.encode_text_calls == 1
    assert model.clip_model.encode_text_calls == 1


def test_categories_reload_clears_text_feature_cache():
    with tempfile.TemporaryDirectory() as temp_dir:
        model, _fake_clip = create_interrogator()
        model.content_dir = temp_dir
        model.text_feature_cache["stale"] = torch.tensor([[1.0, 2.0]])

        with open(f"{temp_dir}/artists.txt", "w", encoding="utf8") as file:
            file.write("artist one\nartist two\n")

        categories = model.categories()

    assert [category.name for category in categories] == ["artists"]
    assert len(model.text_feature_cache) == 0
