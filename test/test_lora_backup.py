import os
import sys


lora_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "extensions-builtin", "Lora"))
if lora_path not in sys.path:
    sys.path.insert(0, lora_path)

import lora_backup  # noqa: E402


class FakeTensor:
    def __init__(self, numel, element_size):
        self._numel = numel
        self._element_size = element_size

    def numel(self):
        return self._numel

    def element_size(self):
        return self._element_size


class FakeLayer:
    pass


def test_estimate_backup_bytes_counts_nested_tensor_backups():
    backup = (FakeTensor(10, 2), [FakeTensor(5, 4), None])

    assert lora_backup.estimate_backup_bytes(backup) == 40


def test_tracked_backup_summary_counts_layers_with_backup_tensors():
    layer = FakeLayer()
    empty_layer = FakeLayer()
    layer.network_weights_backup = FakeTensor(16, 2)
    layer.network_bias_backup = FakeTensor(4, 4)

    lora_backup.track_layer(layer)
    lora_backup.track_layer(empty_layer)

    try:
        summary = lora_backup.summarize_tracked_backups()

        assert summary["layers"] >= 1
        assert summary["bytes"] >= 48
    finally:
        lora_backup.forget_layer(layer)
        lora_backup.forget_layer(empty_layer)


def test_forget_layer_removes_layer_from_backup_summary():
    layer = FakeLayer()
    layer.network_weights_backup = FakeTensor(8, 2)

    lora_backup.track_layer(layer)
    lora_backup.forget_layer(layer)

    assert layer not in lora_backup.iter_tracked_layers()
