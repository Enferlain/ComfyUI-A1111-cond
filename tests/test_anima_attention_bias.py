import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.anima_attention_bias import biased_attention, temporary_attention_bias


class TestExperimentalAttentionBias(unittest.TestCase):
    def test_probability_weighting_is_positive_only_in_both_cfg_orders(self):
        for groups in ([0, 1], [1, 0]):
            for batch_per_group in (1, 2):
                batch = len(groups) * batch_per_group
                q = torch.zeros(batch, 1, 1, 1)
                k = torch.zeros(batch, 2, 1, 1)
                v = torch.tensor([1., 0.]).reshape(1, 2, 1, 1).repeat(batch, 1, 1, 1)
                for weight in (2., .5):
                    output = biased_attention(Mock(), [weight, 1])(q, k, v, {"cond_or_uncond": groups})
                    expected = torch.tensor([weight / (weight + 1) if group == 0 else .5 for group in groups]).repeat_interleave(batch_per_group)
                    self.assertTrue(torch.allclose(output.flatten(), expected))

    def test_neutral_weights_and_negative_only_batches_preserve_backend(self):
        original = Mock(return_value="unchanged")
        q = torch.zeros(1, 1, 1, 1)
        for weights, options in (([1], {}), ([2], {"cond_or_uncond": [1]})):
            self.assertEqual(biased_attention(original, weights)(q, q, q, options), "unchanged")
        self.assertEqual(original.call_count, 2)

    def test_missing_cfg_groups_and_mismatched_length_are_rejected(self):
        q = torch.zeros(1, 1, 1, 1)
        op = biased_attention(Mock(), [2, 1])
        for options in ({}, {"cond_or_uncond": [0]}, {"cond_or_uncond": [2]}):
            with self.assertRaises(ValueError):
                op(q, q, q, options)
        for weights in ([0], [-1], [float("nan")]):
            with self.assertRaises(ValueError):
                biased_attention(Mock(), weights)

    def test_temporary_override_is_restored_after_error(self):
        original = Mock()
        attention = SimpleNamespace(attn_op=original)
        model = SimpleNamespace(model=SimpleNamespace(diffusion_model=SimpleNamespace(
            blocks=[SimpleNamespace(cross_attn=attention)])))
        with self.assertRaises(RuntimeError):
            with temporary_attention_bias(model, [2]):
                self.assertIsNot(attention.attn_op, original)
                raise RuntimeError("probe")
        self.assertIs(attention.attn_op, original)


if __name__ == "__main__":
    unittest.main()
