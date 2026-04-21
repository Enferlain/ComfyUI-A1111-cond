import unittest
import sys
import tempfile
from pathlib import Path

# Add the project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from api.autocomplete import TagDatabase, WildcardDatabase


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


if __name__ == "__main__":
    unittest.main()
