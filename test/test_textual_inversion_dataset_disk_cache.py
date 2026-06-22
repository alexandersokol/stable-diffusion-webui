import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from PIL import Image


fake_images_module = ModuleType("modules.images")
fake_images_module.read = Image.open
sys.modules.setdefault("modules.images", fake_images_module)

from modules.textual_inversion import dataset as dataset_module


class FakeModel:
    def __init__(self):
        self.sample_calls = 0

    def encode_first_stage(self, image):
        return torch.ones((1, 4, 2, 2), dtype=torch.float32)

    def get_first_stage_encoding(self, latent_dist):
        self.sample_calls += 1
        return latent_dist + self.sample_calls


class FakeCondModel:
    def __call__(self, texts):
        return torch.tensor([[4.0, 5.0]], dtype=torch.float32)


@pytest.fixture(autouse=True)
def dataset_shared_state(monkeypatch):
    monkeypatch.setattr(dataset_module.shared, "opts", SimpleNamespace(
        dataset_filename_word_regex="",
        dataset_filename_join_string=" ",
        training_dataset_latent_cache_mode="memory",
    ))
    monkeypatch.setattr(dataset_module.shared, "state", SimpleNamespace(interrupted=False))


def expected_latent(offset=1.0):
    return torch.ones((4, 2, 2), dtype=torch.float32) + offset


def expected_random_latent(offset):
    return torch.ones((1, 4, 2, 2), dtype=torch.float32) + offset


def write_dataset_files(tmp_path, rgba=False):
    data_root = tmp_path / "images"
    data_root.mkdir()
    template_file = tmp_path / "template.txt"
    template_file.write_text("a [name] token with [filewords]\n", encoding="utf8")

    if rgba:
        image = Image.new("RGBA", (4, 4), (8, 16, 24, 255))
        for x in range(image.width):
            for y in range(image.height):
                image.putpixel((x, y), (8, 16, 24, 64 + x * 40 + y * 8))
    else:
        image = Image.new("RGB", (4, 4), (8, 16, 24))

    image.save(data_root / "sample.png")
    return data_root, template_file


def build_dataset(tmp_path, **kwargs):
    data_root, template_file = write_dataset_files(tmp_path, rgba=kwargs.pop("rgba", False))
    model = kwargs.pop("model", FakeModel())
    return dataset_module.PersonalizedBase(
        data_root=str(data_root),
        width=4,
        height=4,
        repeats=1,
        flip_p=0.0,
        placeholder_token="embedding",
        model=model,
        cond_model=FakeCondModel(),
        device="cpu",
        template_file=str(template_file),
        batch_size=1,
        gradient_step=1,
        **kwargs,
    )


def test_memory_mode_keeps_latent_sample_on_dataset_entry(tmp_path):
    ds = build_dataset(tmp_path, cache_mode="memory")

    assert ds.dataset[0].latent_sample is not None
    assert ds[0].latent_sample is ds.dataset[0].latent_sample


def test_disk_mode_spills_latent_sample_and_hydrates_copy(tmp_path):
    ds = build_dataset(tmp_path, cache_mode="disk", cache_dir=tmp_path / "cache")
    stored_entry = ds.dataset[0]

    assert stored_entry.latent_sample is None
    assert stored_entry.cache_file is not None
    assert stored_entry.cache_file.exists()

    loaded_entry = ds[0]

    assert loaded_entry is not stored_entry
    assert torch.equal(loaded_entry.latent_sample, expected_latent())
    assert stored_entry.latent_sample is None


def test_disk_mode_spills_conditioning_and_weights(tmp_path):
    ds = build_dataset(tmp_path, cache_mode="disk", cache_dir=tmp_path / "cache", include_cond=True, use_weight=True, rgba=True)
    stored_entry = ds.dataset[0]

    assert stored_entry.cond is None
    assert stored_entry.weight is None

    loaded_entry = ds[0]

    assert torch.equal(loaded_entry.cond, torch.tensor([4.0, 5.0]))
    assert loaded_entry.weight is not None
    assert loaded_entry.weight.shape == expected_latent().shape


def test_disk_mode_random_sampling_loads_distribution_without_retaining_it(tmp_path, monkeypatch):
    fake_model = FakeModel()
    monkeypatch.setattr(type(dataset_module.shared), "sd_model", property(lambda self: fake_model))
    ds = build_dataset(
        tmp_path,
        cache_mode="disk",
        cache_dir=tmp_path / "cache",
        latent_sampling_method="random",
        model=fake_model,
    )
    stored_entry = ds.dataset[0]

    assert stored_entry.latent_dist is None

    first = ds[0]
    second = ds[0]

    assert torch.equal(first.latent_sample, expected_random_latent(2.0))
    assert torch.equal(second.latent_sample, expected_random_latent(3.0))
    assert stored_entry.latent_dist is None
