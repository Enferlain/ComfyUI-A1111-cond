import unittest
import torch

import comfy_stubs  # noqa: F401 - install isolated ComfyUI dependencies
from A1111_Prompt_Node.encoding.clip import CLIPEncoder

encoder = CLIPEncoder(("l",))
extract_weights = encoder._extract_weights
apply_direct_scaling = encoder._apply_direct_scaling


class TestNodesLogic(unittest.TestCase):
    def test_extract_weights(self):
        tokens = [[(101, 1.0), (102, 1.2), (103, 0.8)], [(201, 1.5), (202, 1.0)]]
        weights = extract_weights(tokens)
        self.assertEqual(len(weights), 2)
        self.assertEqual(weights[0], [1.0, 1.2, 0.8])
        self.assertEqual(weights[1], [1.5, 1.0])

    def test_apply_direct_scaling_basic(self):
        cond = torch.ones((1, 3, 4))
        weights = [[1.0, 2.0, 0.5]]
        scaled = apply_direct_scaling(cond, weights, normalization=False)
        self.assertTrue(torch.allclose(scaled[0, 0], torch.ones(4) * 1.0))
        self.assertTrue(torch.allclose(scaled[0, 1], torch.ones(4) * 2.0))
        self.assertTrue(torch.allclose(scaled[0, 2], torch.ones(4) * 0.5))

    def test_apply_direct_scaling_with_break(self):
        cond = torch.ones((1, 4, 4))
        weights = [[1.2, 1.0], [0.8, 1.0]]
        scaled = apply_direct_scaling(cond, weights, normalization=False)
        expected_weights = [1.2, 1.0, 0.8, 1.0]
        for i, w in enumerate(expected_weights):
            self.assertTrue(torch.allclose(scaled[0, i], torch.ones(4) * w))

    def test_normalization_mean_preservation(self):
        cond = torch.randn((1, 10, 64))
        original_mean = cond.mean().item()
        weights = [[2.0] * 10]

        scaled_no_norm = apply_direct_scaling(cond, weights, normalization=False)
        self.assertAlmostEqual(
            scaled_no_norm.mean().item(), original_mean * 2, places=4
        )

        scaled_norm = apply_direct_scaling(cond, weights, normalization=True)
        self.assertAlmostEqual(scaled_norm.mean().item(), original_mean, places=4)


if __name__ == "__main__":
    unittest.main()
