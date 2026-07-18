import unittest
import csv
import sys
import tempfile
from pathlib import Path

# Add the project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from api.autocomplete import TagDatabase, WildcardDatabase


DATA_DIR = Path(__file__).parent.parent / "data" / "tags"


def load_rows_from_real_csv(filename, tag_names):
    wanted = set(tag_names)
    rows = {}
    with (DATA_DIR / filename).open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) >= 3 and row[0] in wanted:
                rows[row[0]] = row
                if len(rows) == len(wanted):
                    break
    missing = wanted.difference(rows)
    if missing:
        raise AssertionError(f"Missing matrix rows in {filename}: {sorted(missing)}")
    return [rows[name] for name in tag_names]


class TestAutocomplete(unittest.TestCase):
    def setUp(self):
        self.db = TagDatabase()
        self.mock_csv = Path(__file__).parent / "mock_tags.csv"
        self.db.load_csv(self.mock_csv)

    def test_prefix_search(self):
        # Search for "1g" -> should find "1girl"
        results = self.db.search("1g")
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["name"], "1girl")

    def test_alias_matching(self):
        # Search for "sole_female" -> should find "1girl" (via alias)
        results = self.db.search("sole_female")
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["name"], "1girl")
        self.assertEqual(results[0]["matched_alias"], "sole_female")

    def test_sorting_by_count(self):
        # "1girl" has 6M, "1boy" has 1M
        # Searching for "1" should return both, girl first
        results = self.db.search("1")
        names = [r["name"] for r in results]
        self.assertIn("1girl", names)
        self.assertIn("1boy", names)
        # Verify 1girl appears before 1boy due to count
        girl_idx = names.index("1girl")
        boy_idx = names.index("1boy")
        self.assertLess(girl_idx, boy_idx)

    def test_tag_types(self):
        # artist_name is type 1
        results = self.db.search("artist_name")
        self.assertEqual(results[0]["type"], 1)

        # character_name is type 4
        results = self.db.search("character_name")
        self.assertEqual(results[0]["type"], 4)

    def test_empty_query(self):
        results = self.db.search("")
        self.assertEqual(len(results), 0)

    def test_long_query(self):
        # Even if the API handles it, the search method should be robust
        long_query = "a" * 300
        results = self.db.search(long_query)
        self.assertEqual(len(results), 0)

    def test_search_coerces_unusual_inputs(self):
        self.assertEqual(self.db.search(["1girl"]), [])
        self.assertEqual(self.db.search("1girl", limit="bad"), self.db.search("1girl"))
        self.assertEqual(self.db.search("1girl", limit=-5), [])

    def test_short_queries_keep_contains_fallback(self):
        results = self.db.search("rl")
        self.assertEqual(results[0]["name"], "1girl")

    def test_long_queries_keep_contains_fallback(self):
        results = self.db.search("ist_")
        self.assertEqual(results[0]["name"], "artist_name")

    def test_substring_query_finds_middle_of_tag(self):
        results = self.db.search("last_name")
        self.assertEqual(results[0]["name"], "first_name_last_name")

    def test_contains_fallback_can_be_disabled(self):
        self.assertEqual(self.db.search("ist_", contains_fallback=False), [])


class TestAutocompleteUseCaseMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.matrix_csv = Path(cls.temp_dir.name) / "matrix_tags.csv"

        rows = []
        rows.extend(
            load_rows_from_real_csv(
                "danbooru.csv",
                [
                    "1girl",
                    "looking_at_viewer",
                    "open_mouth",
                    "short_hair",
                    "twintails",
                    "school_uniform",
                    "depth_of_field",
                    "holding_hands",
                ],
            )
        )
        rows.extend(
            load_rows_from_real_csv(
                "e621.csv",
                [
                    "anthro",
                    "hi_res",
                    "male",
                    "genitals",
                    "clothing",
                    "hair",
                ],
            )
        )

        with cls.matrix_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerows(rows)

        cls.db = TagDatabase()
        cls.db.load_csv(cls.matrix_csv)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def assertPopupContains(self, query, expected_name, limit=20, search_aliases=True):
        results = self.db.search(query, limit=limit, search_aliases=search_aliases)
        names = [result["name"] for result in results]
        self.assertIn(
            expected_name,
            names,
            msg=f"{expected_name!r} not found for query {query!r}; got {names}",
        )

    def test_likely_typing_prefixes_show_expected_tags(self):
        cases = [
            ("1g", "1girl"),
            ("look", "looking_at_viewer"),
            ("open", "open_mouth"),
            ("short", "short_hair"),
            ("twin", "twintails"),
            ("school", "school_uniform"),
            ("anth", "anthro"),
            ("gen", "genitals"),
            ("cloth", "clothing"),
        ]
        for query, expected_name in cases:
            with self.subTest(query=query, expected_name=expected_name):
                self.assertPopupContains(query, expected_name)

    def test_likely_typing_aliases_show_canonical_tags(self):
        cases = [
            ("sole_female", "1girl"),
            ("mouth_open", "open_mouth"),
            ("short-hair", "short_hair"),
            ("high_res", "hi_res"),
            ("1boy", "male"),
            ("clothes", "clothing"),
        ]
        for query, expected_name in cases:
            with self.subTest(query=query, expected_name=expected_name):
                self.assertPopupContains(query, expected_name)

    def test_likely_typing_middle_fragments_show_expected_tags(self):
        cases = [
            ("at_viewer", "looking_at_viewer"),
            ("mouth", "open_mouth"),
            ("uniform", "school_uniform"),
            ("of_field", "depth_of_field"),
            ("hands", "holding_hands"),
            ("itals", "genitals"),
        ]
        for query, expected_name in cases:
            with self.subTest(query=query, expected_name=expected_name):
                self.assertPopupContains(query, expected_name)

    def test_alias_toggle_removes_alias_only_matches(self):
        results = self.db.search("mouth_open", search_aliases=False)
        self.assertNotIn("open_mouth", [result["name"] for result in results])


class TestWildcardAutocomplete(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.wildcards_dir = Path(self.temp_dir.name)
        (self.wildcards_dir / "colors.txt").write_text("red\nblue\n", encoding="utf-8")
        nested_dir = self.wildcards_dir / "characters"
        nested_dir.mkdir(parents=True, exist_ok=True)
        (nested_dir / "anime.txt").write_text("heroine\n", encoding="utf-8")
        accessories_dir = nested_dir / "accessories"
        accessories_dir.mkdir(parents=True, exist_ok=True)
        (accessories_dir / "hats.txt").write_text("top hat\n", encoding="utf-8")
        self.db = WildcardDatabase(self.wildcards_dir)
        self.db.load()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_wildcard_empty_query_returns_entries(self):
        results = self.db.search("__", limit=10)
        names = [result["name"] for result in results]
        self.assertIn("colors", names)
        self.assertIn("characters/anime", names)

    def test_wildcard_prefix_search(self):
        results = self.db.search("__char", limit=10)
        self.assertEqual(results[0]["name"], "characters")
        self.assertEqual(results[0]["completion"], "__characters__")
        self.assertEqual(results[0]["kind"], "wildcard_folder")

    def test_nested_wildcard_completion_uses_leaf_name(self):
        results = self.db.search("__hats", limit=10)
        self.assertEqual(results[0]["name"], "characters/accessories/hats")
        self.assertEqual(results[0]["leaf_name"], "hats")
        self.assertEqual(results[0]["completion"], "__hats__")

    def test_duplicate_leaf_completion_uses_full_path(self):
        (self.wildcards_dir / "props").mkdir(parents=True, exist_ok=True)
        (self.wildcards_dir / "props" / "hats.txt").write_text("helmet\n", encoding="utf-8")
        self.db.load()

        results = self.db.search("__hats", limit=10)
        completions = {result["name"]: result["completion"] for result in results}
        self.assertEqual(
            completions["characters/accessories/hats"],
            "__characters/accessories/hats__",
        )
        self.assertEqual(completions["props/hats"], "__props/hats__")

    def test_wildcard_contents_search(self):
        results = self.db.get_contents("characters/anime", limit=10)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "heroine")
        self.assertEqual(results[0]["kind"], "wildcard_content")
        self.assertEqual(results[0]["meta"], "characters/anime")

    def test_folder_contents_search_returns_descendants(self):
        results = self.db.get_contents("characters", limit=10)
        names = [result["name"] for result in results]
        self.assertIn("characters/accessories", names)
        self.assertIn("characters/anime", names)
        folder_entry = next(
            result for result in results if result["name"] == "characters/accessories"
        )
        self.assertEqual(folder_entry["kind"], "wildcard_folder")

    def test_wildcard_limits_are_coerced(self):
        self.assertEqual(self.db.search("__", limit=-1), [])
        self.assertEqual(
            self.db.get_contents("characters", limit="bad")[0]["name"],
            "characters/accessories",
        )


if __name__ == "__main__":
    unittest.main()
