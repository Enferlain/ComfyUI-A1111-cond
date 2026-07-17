import sys
import unittest
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from metadata import (
    METADATA_KEY,
    RESOLVED_PROMPT_EXPANDED_PROPERTY,
    RESOLVED_PROMPT_PROPERTY,
    RESOLVED_PROMPT_SOURCE_PROPERTY,
    record_prompt_metadata,
)


class TestPromptMetadata(unittest.TestCase):
    def test_records_original_and_resolved_prompt_by_node_id(self):
        extra_pnginfo = {}

        record_prompt_metadata(
            extra_pnginfo,
            node_id="63",
            node_type="A1111Prompt",
            original_prompt="portrait __subject__",
            resolved_prompt="portrait cat",
        )

        entry = extra_pnginfo[METADATA_KEY]["nodes"]["63"]
        self.assertEqual(entry["node_id"], "63")
        self.assertEqual(entry["type"], "A1111Prompt")
        self.assertEqual(entry["original_prompt"], "portrait __subject__")
        self.assertEqual(entry["resolved_prompt"], "portrait cat")
        self.assertTrue(entry["changed"])

    def test_ignores_missing_pnginfo_or_node_id(self):
        record_prompt_metadata(None, "63", "A1111Prompt", "a", "b")

        extra_pnginfo = {}
        record_prompt_metadata(extra_pnginfo, None, "A1111Prompt", "a", "b")
        self.assertEqual(extra_pnginfo, {})

    def test_preserves_existing_prompt_entries(self):
        extra_pnginfo = {}

        record_prompt_metadata(extra_pnginfo, "63", "A1111Prompt", "a", "b")
        record_prompt_metadata(extra_pnginfo, "64", "A1111PromptNegative", "c", "c")

        nodes = extra_pnginfo[METADATA_KEY]["nodes"]
        self.assertEqual(set(nodes), {"63", "64"})
        self.assertFalse(nodes["64"]["changed"])

    def test_writes_resolved_prompt_to_embedded_workflow_node_properties(self):
        extra_pnginfo = {
            "workflow": {
                "nodes": [
                    {"id": 12, "type": "OtherNode", "properties": {}},
                    {"id": 63, "type": "A1111Prompt", "properties": {}},
                ]
            }
        }

        record_prompt_metadata(
            extra_pnginfo,
            node_id="63",
            node_type="A1111Prompt",
            original_prompt="portrait __subject__",
            resolved_prompt="portrait cat",
        )

        properties = extra_pnginfo["workflow"]["nodes"][1]["properties"]
        self.assertEqual(properties[RESOLVED_PROMPT_PROPERTY], "portrait cat")
        self.assertEqual(properties[RESOLVED_PROMPT_SOURCE_PROPERTY], "portrait __subject__")
        self.assertFalse(properties[RESOLVED_PROMPT_EXPANDED_PROPERTY])

    def test_removes_stale_workflow_preview_when_prompt_is_unchanged(self):
        extra_pnginfo = {
            "workflow": {
                "nodes": [
                    {
                        "id": "63",
                        "type": "A1111Prompt",
                        "properties": {
                            RESOLVED_PROMPT_PROPERTY: "old resolved",
                            RESOLVED_PROMPT_EXPANDED_PROPERTY: True,
                        },
                    }
                ]
            }
        }

        record_prompt_metadata(
            extra_pnginfo,
            node_id="63",
            node_type="A1111Prompt",
            original_prompt="same prompt",
            resolved_prompt="same prompt",
        )

        properties = extra_pnginfo["workflow"]["nodes"][0]["properties"]
        self.assertNotIn(RESOLVED_PROMPT_PROPERTY, properties)
        self.assertNotIn(RESOLVED_PROMPT_EXPANDED_PROPERTY, properties)
        self.assertNotIn(RESOLVED_PROMPT_SOURCE_PROPERTY, properties)


if __name__ == "__main__":
    unittest.main()
