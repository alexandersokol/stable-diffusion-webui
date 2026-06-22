import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import skimage


def load_outpainting_module_for_test():
    import modules as modules_package

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "outpainting_mk_2.py"
    stubs = {
        "gradio": types.ModuleType("gradio"),
        "modules.scripts": types.SimpleNamespace(Script=object),
        "modules.images": types.ModuleType("modules.images"),
        "modules.processing": types.SimpleNamespace(Processed=object, process_images=lambda *args, **kwargs: None),
        "modules.shared": types.SimpleNamespace(opts=types.SimpleNamespace(), state=types.SimpleNamespace()),
    }

    previous = {name: sys.modules.get(name) for name in stubs}
    previous_package_attrs = {}
    for name in stubs:
        if not name.startswith("modules."):
            continue

        attr_name = name.split(".", 1)[1]
        previous_package_attrs[attr_name] = (
            hasattr(modules_package, attr_name),
            getattr(modules_package, attr_name, None),
        )

    sys.modules.update(stubs)
    for name, module in stubs.items():
        if name.startswith("modules."):
            setattr(modules_package, name.split(".", 1)[1], module)

    try:
        spec = importlib.util.spec_from_file_location("_test_outpainting_mk_2", script_path)
        outpainting = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(outpainting)
        return outpainting
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

        for attr_name, (had_attr, value) in previous_package_attrs.items():
            if had_attr:
                setattr(modules_package, attr_name, value)
            elif hasattr(modules_package, attr_name):
                delattr(modules_package, attr_name)


def legacy_get_matched_noise(_np_src_image, np_mask_rgb, noise_q=1, color_variation=0.05):
    def _fft2(data):
        if data.ndim > 2:
            out_fft = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.complex128)
            for c in range(data.shape[2]):
                c_data = data[:, :, c]
                out_fft[:, :, c] = np.fft.fft2(np.fft.fftshift(c_data), norm="ortho")
                out_fft[:, :, c] = np.fft.ifftshift(out_fft[:, :, c])
        else:
            out_fft = np.zeros((data.shape[0], data.shape[1]), dtype=np.complex128)
            out_fft[:, :] = np.fft.fft2(np.fft.fftshift(data), norm="ortho")
            out_fft[:, :] = np.fft.ifftshift(out_fft[:, :])

        return out_fft

    def _ifft2(data):
        if data.ndim > 2:
            out_ifft = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.complex128)
            for c in range(data.shape[2]):
                c_data = data[:, :, c]
                out_ifft[:, :, c] = np.fft.ifft2(np.fft.fftshift(c_data), norm="ortho")
                out_ifft[:, :, c] = np.fft.ifftshift(out_ifft[:, :, c])
        else:
            out_ifft = np.zeros((data.shape[0], data.shape[1]), dtype=np.complex128)
            out_ifft[:, :] = np.fft.ifft2(np.fft.fftshift(data), norm="ortho")
            out_ifft[:, :] = np.fft.ifftshift(out_ifft[:, :])

        return out_ifft

    def _get_gaussian_window(width, height, std=3.14, mode=0):
        window_scale_x = float(width / min(width, height))
        window_scale_y = float(height / min(width, height))

        window = np.zeros((width, height))
        x = (np.arange(width) / width * 2. - 1.) * window_scale_x
        for y in range(height):
            fy = (y / height * 2. - 1.) * window_scale_y
            if mode == 0:
                window[:, y] = np.exp(-(x ** 2 + fy ** 2) * std)
            else:
                window[:, y] = (1 / ((x ** 2 + 1.) * (fy ** 2 + 1.))) ** (std / 3.14)

        return window

    def _get_masked_window_rgb(np_mask_grey, hardness=1.):
        np_mask_rgb = np.zeros((np_mask_grey.shape[0], np_mask_grey.shape[1], 3))
        if hardness != 1.:
            hardened = np_mask_grey[:] ** hardness
        else:
            hardened = np_mask_grey[:]
        for c in range(3):
            np_mask_rgb[:, :, c] = hardened[:]
        return np_mask_rgb

    width = _np_src_image.shape[0]
    height = _np_src_image.shape[1]
    num_channels = _np_src_image.shape[2]

    _np_src_image[:] * (1. - np_mask_rgb)
    np_mask_grey = (np.sum(np_mask_rgb, axis=2) / 3.)
    img_mask = np_mask_grey > 1e-6
    ref_mask = np_mask_grey < 1e-3

    windowed_image = _np_src_image * (1. - _get_masked_window_rgb(np_mask_grey))
    windowed_image /= np.max(windowed_image)
    windowed_image += np.average(_np_src_image) * np_mask_rgb

    src_fft = _fft2(windowed_image)
    src_dist = np.absolute(src_fft)
    src_phase = src_fft / src_dist

    rng = np.random.default_rng(0)

    noise_window = _get_gaussian_window(width, height, mode=1)
    noise_rgb = rng.random((width, height, num_channels))
    noise_grey = (np.sum(noise_rgb, axis=2) / 3.)
    noise_rgb *= color_variation
    for c in range(num_channels):
        noise_rgb[:, :, c] += (1. - color_variation) * noise_grey

    noise_fft = _fft2(noise_rgb)
    for c in range(num_channels):
        noise_fft[:, :, c] *= noise_window
    noise_rgb = np.real(_ifft2(noise_fft))
    shaped_noise_fft = _fft2(noise_rgb)
    shaped_noise_fft[:, :, :] = np.absolute(shaped_noise_fft[:, :, :]) ** 2 * (src_dist ** noise_q) * src_phase

    brightness_variation = 0.
    contrast_adjusted_np_src = _np_src_image[:] * (brightness_variation + 1.) - brightness_variation * 2.

    shaped_noise = np.real(_ifft2(shaped_noise_fft))
    shaped_noise -= np.min(shaped_noise)
    shaped_noise /= np.max(shaped_noise)
    shaped_noise[img_mask, :] = skimage.exposure.match_histograms(shaped_noise[img_mask, :] ** 1., contrast_adjusted_np_src[ref_mask, :], channel_axis=1)
    shaped_noise = _np_src_image[:] * (1. - np_mask_rgb) + shaped_noise * np_mask_rgb

    matched_noise = shaped_noise[:]

    return np.clip(matched_noise, 0., 1.)


def make_test_image_and_mask():
    rng = np.random.default_rng(12345)
    src = rng.random((8, 10, 3))
    mask = np.zeros_like(src)
    mask[:, :4, :] = 1.0
    mask[2:6, 4:6, :] = 0.5

    return src, mask


def test_get_matched_noise_matches_legacy_output_and_uses_stacked_fft(monkeypatch):
    outpainting = load_outpainting_module_for_test()
    src, mask = make_test_image_and_mask()
    expected = legacy_get_matched_noise(src.copy(), mask.copy(), noise_q=1.25, color_variation=0.35)
    original_fft2 = outpainting.np.fft.fft2
    original_ifft2 = outpainting.np.fft.ifft2
    fft2_shapes = []
    ifft2_shapes = []

    def recording_fft2(data, *args, **kwargs):
        fft2_shapes.append(data.shape)
        return original_fft2(data, *args, **kwargs)

    def recording_ifft2(data, *args, **kwargs):
        ifft2_shapes.append(data.shape)
        return original_ifft2(data, *args, **kwargs)

    monkeypatch.setattr(outpainting.np.fft, "fft2", recording_fft2)
    monkeypatch.setattr(outpainting.np.fft, "ifft2", recording_ifft2)

    actual = outpainting.get_matched_noise(src.copy(), mask.copy(), noise_q=1.25, color_variation=0.35)

    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert fft2_shapes == [(8, 10, 3), (8, 10, 3), (8, 10, 3)]
    assert ifft2_shapes == [(8, 10, 3), (8, 10, 3)]
