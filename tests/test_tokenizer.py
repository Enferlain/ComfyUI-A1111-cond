import unittest
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

# Mock ComfyUI dependencies before importing logic
sys.modules["aiohttp"] = MagicMock()
sys.modules["server"] = MagicMock()
sys.modules["comfy"] = MagicMock()
sys.modules["comfy.sd1_clip"] = MagicMock()

from api.tokenize import strip_a1111_syntax


class TestTokenizer(unittest.TestCase):
    def test_emphasis_stripping(self):
        # (word:1.2) -> word
        self.assertEqual(strip_a1111_syntax("(dog:1.3)"), "dog")
        # (word) -> word
        self.assertEqual(strip_a1111_syntax("(cat)"), "cat")
        # multiple emphasis
        self.assertEqual(
            strip_a1111_syntax("a (beautiful:1.2) (cat:1.5)"), "a beautiful cat"
        )

    def test_bracket_emphasis(self):
        # [word] -> word
        self.assertEqual(strip_a1111_syntax("[bird]"), "bird")
        # [word:0.5] -> word
        self.assertEqual(strip_a1111_syntax("[fish:0.5]"), "fish")

    def test_scheduling_stripping(self):
        # [from:to:when] -> keeps longest
        self.assertEqual(strip_a1111_syntax("[mountain:ocean:0.5]"), "mountain")
        self.assertEqual(
            strip_a1111_syntax("[forest:very_large_city:0.5]"), "very_large_city"
        )
        # [add:when] -> add
        self.assertEqual(strip_a1111_syntax("[glasses:10]"), "glasses")
        # [remove::when] -> remove
        self.assertEqual(strip_a1111_syntax("[hat::15]"), "hat")

    def test_alternation_stripping(self):
        # [A|B|C] -> keeps longest
        self.assertEqual(strip_a1111_syntax("[red|blue|green]"), "green")
        self.assertEqual(strip_a1111_syntax("[a|bc|d]"), "bc")

    def test_scheduled_alternation_stripping(self):
        # [A|B:0.5] -> keeps longest
        self.assertEqual(strip_a1111_syntax("[apple|banana:0.5]"), "banana")
        self.assertEqual(strip_a1111_syntax("[apple|banana::0.5]"), "banana")

    def test_escaped_stripping(self):
        # \( \) -> ( )
        self.assertEqual(strip_a1111_syntax(r"\(escaped\)"), "(escaped)")
        self.assertEqual(strip_a1111_syntax(r"\[bracket\]"), "[bracket]")

    def test_nested_stripping(self):
        # [(red:1.2):[blue|green]:0.5]
        # Inner [blue|green] -> green (len 5 vs blue len 4)
        # Then [(red:1.2):green:0.5] -> red (len 3+tags removal) vs green (len 5)
        # Result should be "green" because "red" is len 3 and "green" is len 5
        self.assertEqual(strip_a1111_syntax("[(red:1.2):green:0.5]"), "green")


if __name__ == "__main__":
    unittest.main()
