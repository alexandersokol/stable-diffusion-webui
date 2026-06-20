import itertools


def format_bytes(size):
    size = int(size or 0)
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)

    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"

        value /= 1024.0


def estimate_tensor_bytes(tensor):
    if tensor is None:
        return 0

    try:
        return int(tensor.numel()) * int(tensor.element_size())
    except Exception:
        return 0


def _tensor_storage_key(tensor):
    try:
        storage = tensor.untyped_storage()
        return ("storage", storage.data_ptr(), tensor.device, storage.nbytes())
    except Exception:
        return ("object", id(tensor))


def _call_tensor_iter(model, name):
    fn = getattr(model, name, None)
    if fn is None:
        return ()

    try:
        return fn()
    except TypeError:
        return fn(recurse=True)
    except Exception:
        return ()


def estimate_model_bytes(model):
    seen = set()
    total = 0

    for tensor in itertools.chain(_call_tensor_iter(model, "parameters"), _call_tensor_iter(model, "buffers")):
        key = _tensor_storage_key(tensor)
        if key in seen:
            continue

        seen.add(key)
        total += estimate_tensor_bytes(tensor)

    return total


def ensure_model_cache_info(model):
    if model is None:
        return 0

    size = getattr(model, "sd_model_cache_bytes", None)
    if size is None:
        size = estimate_model_bytes(model)
        setattr(model, "sd_model_cache_bytes", size)

    return int(size or 0)


def total_model_cache_bytes(models):
    return sum(ensure_model_cache_info(model) for model in models)


def select_models_to_evict(models, protected_models=None, max_count=0, max_bytes=0):
    protected_ids = {id(model) for model in protected_models or () if model is not None}
    remaining = list(models)
    evicted = []

    def oldest_evictable_model():
        for model in reversed(remaining):
            if id(model) not in protected_ids:
                return model
        return None

    while max_count > 0 and len(remaining) > max_count:
        model = oldest_evictable_model()
        if model is None:
            break

        remaining.remove(model)
        evicted.append(model)

    if max_bytes > 0:
        total_bytes = total_model_cache_bytes(remaining)
        while total_bytes > max_bytes:
            model = oldest_evictable_model()
            if model is None:
                break

            remaining.remove(model)
            evicted.append(model)
            total_bytes -= ensure_model_cache_info(model)

    return evicted


def can_add_model_to_cache(models, max_count=0, max_bytes=0, estimated_new_bytes=None):
    models = list(models)

    if max_count > 0 and len(models) >= max_count:
        return False

    if max_bytes <= 0:
        return True

    current_bytes = total_model_cache_bytes(models)
    if estimated_new_bytes is None:
        estimated_new_bytes = max((ensure_model_cache_info(model) for model in models), default=0)

    return current_bytes + int(estimated_new_bytes or 0) <= max_bytes
