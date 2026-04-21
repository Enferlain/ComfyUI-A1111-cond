import random
import sys
import tempfile
import unittest
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from parser.wildcards import (
    expand_wildcards,
    get_wildcard_options,
    list_available_wildcards,
)


class TestWildcards(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.wildcards_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_wildcard(self, relative_path: str, contents: str):
        wildcard_file = self.wildcards_dir / relative_path
        wildcard_file.parent.mkdir(parents=True, exist_ok=True)
        wildcard_file.write_text(contents, encoding="utf-8")

    def test_get_wildcard_options_ignores_comments_and_blank_lines(self):
        self.write_wildcard(
            "colors.txt",
            "\n# comment\nred\n\nblue\n  \n# another comment\ngreen\n",
        )

        self.assertEqual(
            get_wildcard_options("colors", wildcards_dir=str(self.wildcards_dir)),
            ["red", "blue", "green"],
        )

    def test_list_available_wildcards_uses_dot_notation(self):
        self.write_wildcard("colors.txt", "red\n")
        self.write_wildcard("characters/anime.txt", "heroine\n")

        self.assertEqual(
            list_available_wildcards(wildcards_dir=str(self.wildcards_dir)),
            ["characters.anime", "colors"],
        )

    def test_expand_wildcards_resolves_nested_entries(self):
        self.write_wildcard("colors.txt", "red\n")
        self.write_wildcard("outfits.txt", "__colors__ dress\n")

        expanded = expand_wildcards(
            "portrait, __outfits__",
            wildcards_dir=str(self.wildcards_dir),
            rng=random.Random(0),
        )

        self.assertEqual(expanded, "portrait, red dress")

    def test_expand_wildcards_supports_nested_directory_names(self):
        self.write_wildcard("characters/anime.txt", "mahou shoujo\n")

        expanded = expand_wildcards(
            "__characters.anime__",
            wildcards_dir=str(self.wildcards_dir),
            rng=random.Random(0),
        )

        self.assertEqual(expanded, "mahou shoujo")

    def test_expand_wildcards_resolves_dynamic_prompt_choices(self):
        expanded = expand_wildcards(
            "wearing {red|blue} dress",
            wildcards_dir=str(self.wildcards_dir),
            rng=random.Random(0),
        )

        self.assertEqual(expanded, "wearing blue dress")

    def test_expand_wildcards_resolves_nested_wildcards_and_dynamic_prompts(self):
        self.write_wildcard(
            "outfits.txt",
            "__tops__{, __accessories__|}\n",
        )
        self.write_wildcard("tops.txt", "shirt\n")
        self.write_wildcard("accessories.txt", "hat\n")

        expanded = expand_wildcards(
            "__outfits__",
            wildcards_dir=str(self.wildcards_dir),
            rng=random.Random(0),
        )

        self.assertEqual(expanded, "shirt, hat")

    def test_expand_wildcards_supports_dynamic_prompt_ranges(self):
        expanded = expand_wildcards(
            "{2$$red|blue|green}",
            wildcards_dir=str(self.wildcards_dir),
            rng=random.Random(0),
        )

        self.assertEqual(expanded, "green, red")

    def test_expand_wildcards_normalizes_spacing_after_empty_dynamic_choice(self):
        self.write_wildcard("colors.txt", "red\n")
        self.write_wildcard("shirt.txt", "{0-0$$__colors__} sleeveless turtleneck\n")

        expanded = expand_wildcards(
            "__shirt__",
            wildcards_dir=str(self.wildcards_dir),
            rng=random.Random(0),
        )

        self.assertEqual(expanded, "sleeveless turtleneck")

    def test_missing_wildcards_are_left_unchanged(self):
        self.assertEqual(
            expand_wildcards(
                "test __missing__",
                wildcards_dir=str(self.wildcards_dir),
                rng=random.Random(0),
            ),
            "test __missing__",
        )

    def test_recursive_loops_do_not_hang(self):
        self.write_wildcard("loop.txt", "__loop__\n")

        self.assertEqual(
            expand_wildcards(
                "__loop__",
                wildcards_dir=str(self.wildcards_dir),
                rng=random.Random(0),
                max_depth=4,
            ),
            "__loop__",
        )


if __name__ == "__main__":
    unittest.main()
