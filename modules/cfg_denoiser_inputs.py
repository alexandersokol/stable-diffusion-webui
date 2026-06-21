import torch


def build_repeat_indexes(repeats, device):
    repeat_counts = torch.as_tensor(repeats, device=device, dtype=torch.long)
    return torch.repeat_interleave(torch.arange(len(repeats), device=device), repeat_counts)


def build_repeated_batch(tensor, indexes):
    if indexes.device != tensor.device:
        indexes = indexes.to(device=tensor.device)
    return tensor.index_select(0, indexes)


def build_cfg_denoiser_inputs(x, sigma, image_cond, image_uncond, repeats, edit_image_cond=None):
    indexes = build_repeat_indexes(repeats, x.device)
    x_in = torch.cat([build_repeated_batch(x, indexes), x])
    sigma_in = torch.cat([build_repeated_batch(sigma, indexes), sigma])
    image_cond_in = torch.cat([build_repeated_batch(image_cond, indexes), image_uncond])

    if edit_image_cond is not None:
        x_in = torch.cat([x_in, x])
        sigma_in = torch.cat([sigma_in, sigma])
        image_cond_in = torch.cat([image_cond_in, edit_image_cond])

    return x_in, sigma_in, image_cond_in
