import unittest
import torch

# We extract the pure logic from prompt_node.py to test it in isolation
# without dealing with ComfyUI's complex package/import system.


def extract_weights(token_batches):
    all_weights = []
    for batch in token_batches:
        if not batch:
            all_weights.append([])
            continue
        batch_weights = [
            t[1] if hasattr(t, "__getitem__") and len(t) > 1 else 1.0 for t in batch
        ]
        all_weights.append(batch_weights)
    return all_weights


def apply_direct_scaling(cond, weights, normalization):
    cond = cond.clone()
    flat_weights = []
    for chunk_weights in weights:
        flat_weights.extend(chunk_weights)

    seq_len = cond.shape[1]
    if len(flat_weights) < seq_len:
        flat_weights = flat_weights + [1.0] * (seq_len - len(flat_weights))
    elif len(flat_weights) > seq_len:
        flat_weights = flat_weights[:seq_len]

    batch_size = cond.shape[0]
    multipliers = torch.tensor(
        [flat_weights] * batch_size, device=cond.device, dtype=cond.dtype
    )

    if normalization:
        original_mean = cond.mean()

    multipliers = multipliers.unsqueeze(-1).expand_as(cond)
    cond = cond * multipliers

    if normalization:
        new_mean = cond.mean()
        if new_mean.abs() > 1e-8:
            scale_factor = original_mean / new_mean
            cond = cond * scale_factor
    return cond


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
