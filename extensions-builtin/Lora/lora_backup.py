from __future__ import annotations

import weakref


tracked_layers = weakref.WeakSet()


def track_layer(layer):
    """Remember a module that owns LoRA backup tensors."""

    tracked_layers.add(layer)


def forget_layer(layer):
    tracked_layers.discard(layer)


def iter_tracked_layers():
    return list(tracked_layers)


def tensor_bytes(tensor):
    if tensor is None:
        return 0

    try:
        return tensor.numel() * tensor.element_size()
    except Exception:
        return 0


def backup_tensors(backup):
    if backup is None:
        return

    if isinstance(backup, (list, tuple)):
        for item in backup:
            yield from backup_tensors(item)
        return

    yield backup


def estimate_backup_bytes(backup):
    return sum(tensor_bytes(tensor) for tensor in backup_tensors(backup))


def estimate_layer_backup_bytes(layer):
    return (
        estimate_backup_bytes(getattr(layer, "network_weights_backup", None)) +
        estimate_backup_bytes(getattr(layer, "network_bias_backup", None))
    )


def layer_has_backup(layer):
    return (
        getattr(layer, "network_weights_backup", None) is not None or
        getattr(layer, "network_bias_backup", None) is not None
    )


def summarize_tracked_backups():
    layers = 0
    bytes_used = 0

    for layer in iter_tracked_layers():
        layer_bytes = estimate_layer_backup_bytes(layer)
        if layer_bytes == 0:
            continue

        layers += 1
        bytes_used += layer_bytes

    return {"layers": layers, "bytes": bytes_used}


def format_bytes(num_bytes):
    if num_bytes < 1024:
        return f"{num_bytes} B"

    units = ["KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        value /= 1024.0
        if abs(value) < 1024.0:
            return f"{value:.1f} {unit}"

    return f"{value:.1f} PB"
