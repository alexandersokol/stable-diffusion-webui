import unittest

import torch

from modules.cfg_denoiser_inputs import build_cfg_denoiser_inputs


def legacy_repeated_input(tensor, repeats):
    return torch.cat([torch.stack([tensor[i] for _ in range(n)]) for i, n in enumerate(repeats)])


class CFGDenoiserInputsTest(unittest.TestCase):
    def test_build_cfg_denoiser_inputs_matches_legacy_normal_ordering(self):
        repeats = [2, 1, 3]
        x = torch.arange(3 * 2 * 2, dtype=torch.float32).reshape(3, 2, 2)
        sigma = torch.tensor([0.3, 0.2, 0.1])
        image_cond = torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4)
        image_uncond = image_cond + 100

        x_in, sigma_in, image_cond_in = build_cfg_denoiser_inputs(x, sigma, image_cond, image_uncond, repeats)

        self.assertTrue(torch.equal(torch.cat([legacy_repeated_input(x, repeats), x]), x_in))
        self.assertTrue(torch.equal(torch.cat([legacy_repeated_input(sigma, repeats), sigma]), sigma_in))
        self.assertTrue(torch.equal(torch.cat([legacy_repeated_input(image_cond, repeats), image_uncond]), image_cond_in))

    def test_build_cfg_denoiser_inputs_matches_legacy_edit_model_ordering(self):
        repeats = [1, 2]
        x = torch.arange(2 * 3, dtype=torch.float32).reshape(2, 3)
        sigma = torch.tensor([0.6, 0.4])
        image_cond = torch.arange(2 * 2, dtype=torch.float32).reshape(2, 2)
        image_uncond = image_cond + 10
        image_edit_uncond = torch.zeros_like(image_cond)

        x_in, sigma_in, image_cond_in = build_cfg_denoiser_inputs(
            x, sigma, image_cond, image_uncond, repeats, edit_image_cond=image_edit_uncond
        )

        self.assertTrue(torch.equal(torch.cat([legacy_repeated_input(x, repeats), x, x]), x_in))
        self.assertTrue(torch.equal(torch.cat([legacy_repeated_input(sigma, repeats), sigma, sigma]), sigma_in))
        self.assertTrue(torch.equal(torch.cat([legacy_repeated_input(image_cond, repeats), image_uncond, image_edit_uncond]), image_cond_in))


if __name__ == "__main__":
    unittest.main()
