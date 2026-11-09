import unittest
from unittest.mock import patch

import torch

import comfy_stubs  # noqa: F401 - install isolated ComfyUI dependencies
from A1111_Prompt_Node.nodes.prompt_node import A1111PromptNode
from A1111_Prompt_Node.encoding.base import EncoderCapabilities, PromptEncoder


class FakeClip:
    pass


class CacheProbeEncoder(PromptEncoder):
    name = "Cache probe"
    capabilities = EncoderCapabilities(breaks=True, scheduling=True, normalization=True)

    def matches(self, clip, streams):
        return True

    def encode(self, clip, prompt_text, normalization=False, debug=False):
        seq_len = 2 if prompt_text == "short" else 5
        cond = torch.ones((1, seq_len, 3))
        pooled = torch.ones((1, 3))
        return cond, {"pooled_output": pooled}


class CacheProbePromptNode(A1111PromptNode):
    def _get_downstream_steps(self, prompt, start_node_id, debug=False):
        return 2


class TestPromptNodeCache(unittest.TestCase):
    def test_scheduled_padding_does_not_mutate_encoded_cache(self):
        node = CacheProbePromptNode()
        clip = FakeClip()

        with patch("A1111_Prompt_Node.nodes.prompt_node.get_encoder", return_value=CacheProbeEncoder()):
            node.encode(clip, "[short:long:0.5]", prompt={}, unique_id="1")

        cached_shapes = {
            key[2]: cond.shape[1]
            for key, (cond, _metadata) in node._encoded_cache.items()
        }
        self.assertEqual(cached_shapes["short"], 2)
        self.assertEqual(cached_shapes["long"], 5)


if __name__ == "__main__":
    unittest.main()
