"""Pure span-mapping checks for the production Anima tokenizer helper."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "encoding"))
from anima_tokenization import join_weighted_segments, weights_for_offsets


class TestExperimentalEmphasis(unittest.TestCase):
    def test_complete_artist_span_includes_at_prefix_and_multiple_words(self):
        text = join_weighted_segments([
            ("portrait, ", 1), ("@artist name", 1.5), (", smile", 1)
        ])
        weights, crossings = weights_for_offsets(
            text, [(0, 8), (8, 9), (10, 11), (11, 17), (18, 22), (22, 23), (24, 29), (0, 0)]
        )
        self.assertEqual(weights, [1, 1, 1.5, 1.5, 1.5, 1, 1, 1])
        self.assertEqual(crossings, [])

    def test_partial_token_weights_are_character_averaged_and_reported(self):
        text = join_weighted_segments([("ar", 2), ("tist", 1)])
        weights, crossings = weights_for_offsets(text, [(0, 6)])
        self.assertAlmostEqual(weights[0], 4 / 3)
        self.assertEqual(crossings, [0])

    def test_whitespace_does_not_dilute_weight_of_content_token(self):
        text = join_weighted_segments([(" ", 1), ("artist", 2)])
        self.assertEqual(weights_for_offsets(text, [(0, 7)])[0], [2])

    def test_special_and_whitespace_only_tokens_are_neutral(self):
        text = join_weighted_segments([(" ", 2)])
        self.assertEqual(weights_for_offsets(text, [(0, 0), (0, 1)])[0], [1, 1])

    def test_unicode_offsets_use_characters(self):
        text = join_weighted_segments([("@猫", 1.5), (",", 1)])
        self.assertEqual(weights_for_offsets(text, [(0, 1), (1, 2), (2, 3)])[0], [1.5, 1.5, 1])

    def test_bad_offsets_and_nonfinite_weights_are_rejected(self):
        with self.assertRaises(ValueError):
            join_weighted_segments([("cat", float("nan"))])
        with self.assertRaises(ValueError):
            weights_for_offsets(join_weighted_segments([("cat", 1)]), [(0, 4)])


if __name__ == "__main__":
    unittest.main()
