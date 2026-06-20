from modules import devices, model_cache_budget


def get_model_object(model_or_descriptor):
    if model_or_descriptor is None:
        return None

    return getattr(model_or_descriptor, "model", model_or_descriptor)


def estimate_cached_model_bytes(model_or_descriptor):
    model = get_model_object(model_or_descriptor)
    return model_cache_budget.estimate_model_bytes(model)


def format_cached_model_size(model_or_descriptor):
    return model_cache_budget.format_bytes(estimate_cached_model_bytes(model_or_descriptor))


def unload_cached_model(model_or_descriptor):
    model = get_model_object(model_or_descriptor)
    if model is None:
        return

    try:
        model.to(device="meta")
    except Exception:
        try:
            model.to(devices.cpu)
        except Exception:
            pass

    devices.torch_gc()
