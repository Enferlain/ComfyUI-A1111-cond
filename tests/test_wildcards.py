import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from parser.wildcards import (
    expand_wildcards,
    expand_wildcards_for_token_count,
    get_wildcard_options,
    list_available_wildcards,
)
from parser import wildcards as wildcards_module


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

    def test_expand_wildcards_resolves_unique_leaf_names(self):
        self.write_wildcard("characters/accessories/hats.txt", "wide brim hat\n")

        expanded = expand_wildcards(
            "__hats__",
            wildcards_dir=str(self.wildcards_dir),
            rng=random.Random(0),
        )

        self.assertEqual(expanded, "wide brim hat")

    def test_unique_leaf_resolution_scans_the_tree_only_once(self):
        self.write_wildcard("nested/first.txt", "one\n")
        self.write_wildcard("nested/second.txt", "two\n")
        self.write_wildcard("nested/third.txt", "three\n")

        with patch.object(
            wildcards_module,
            "_iter_wildcard_files",
            wraps=wildcards_module._iter_wildcard_files,
        ) as iter_files:
            self.assertEqual(
                get_wildcard_options("first", str(self.wildcards_dir)), ["one"]
            )
            self.assertEqual(
                get_wildcard_options("second", str(self.wildcards_dir)), ["two"]
            )
            self.assertEqual(
                get_wildcard_options("third", str(self.wildcards_dir)), ["three"]
            )

        self.assertEqual(iter_files.call_count, 1)

    def test_ambiguous_leaf_wildcards_are_left_unchanged(self):
        self.write_wildcard("characters/hats.txt", "beret\n")
        self.write_wildcard("props/hats.txt", "helmet\n")

        with self.assertLogs("A1111PromptNode", level="WARNING"):
            expanded = expand_wildcards(
                "__hats__",
                wildcards_dir=str(self.wildcards_dir),
                rng=random.Random(0),
            )

        self.assertEqual(expanded, "__hats__")

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

    def test_token_count_expansion_uses_highest_scoring_wildcard_option(self):
        self.write_wildcard("subject.txt", "cat\nvery very long subject\n")

        expanded = expand_wildcards_for_token_count(
            "portrait of __subject__",
            wildcards_dir=str(self.wildcards_dir),
            scorer=lambda value: len(value.split()),
        )

        self.assertEqual(expanded, "portrait of very very long subject")

    def test_token_count_expansion_uses_max_dynamic_prompt_quantity(self):
        expanded = expand_wildcards_for_token_count(
            "{1-2$$red|blue|extremely long green}",
            wildcards_dir=str(self.wildcards_dir),
            scorer=lambda value: len(value.split()),
        )

        self.assertEqual(expanded, "extremely long green, blue")

    def test_token_count_expansion_resolves_nested_wildcards(self):
        self.write_wildcard("colors.txt", "red\nultramarine blue\n")
        self.write_wildcard("outfits.txt", "__colors__ robe\nhat\n")

        expanded = expand_wildcards_for_token_count(
            "__outfits__",
            wildcards_dir=str(self.wildcards_dir),
            scorer=lambda value: len(value.split()),
        )

        self.assertEqual(expanded, "ultramarine blue robe")


if __name__ == "__main__":
    unittest.main()
