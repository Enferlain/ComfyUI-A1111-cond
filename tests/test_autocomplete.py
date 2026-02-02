import unittest
import sys
import os
from pathlib import Path

# Add the project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from api.autocomplete import TagDatabase, TagEntry


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


if __name__ == "__main__":
    unittest.main()
