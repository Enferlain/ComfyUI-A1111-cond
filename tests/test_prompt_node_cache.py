import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

mock_model_management = MagicMock()
mock_model_management.intermediate_device.return_value = "cpu"

mock_comfy = MagicMock()
mock_comfy.model_management = mock_model_management

mock_hooks = MagicMock()


class DummyHook:
    def __init__(self, *args, **kwargs):
        pass

    def clone(self):
        return self


class DummyHookGroup:
    def add(self, hook):
        self.hook = hook


mock_hooks.TransformerOptionsHook = DummyHook
mock_hooks.HookGroup = DummyHookGroup
mock_hooks.EnumHookScope = MagicMock()
mock_hooks.set_hooks_for_conditioning = (
    lambda conditioning, hooks=None, append_hooks=True: conditioning
)
mock_comfy.hooks = mock_hooks

server = MagicMock()
server.PromptServer.instance.routes.post = lambda path: (lambda fn: fn)
server.PromptServer.instance.routes.get = lambda path: (lambda fn: fn)

sys.modules["comfy"] = mock_comfy
sys.modules["comfy.model_management"] = mock_model_management
sys.modules["comfy.hooks"] = mock_hooks
sys.modules["server"] = server

from A1111_Prompt_Node.nodes.prompt_node import A1111PromptNode


class FakeCondStageModel:
    pass


class FakeClip:
    def __init__(self):
        self.cond_stage_model = FakeCondStageModel()


class CacheProbePromptNode(A1111PromptNode):
    def _get_downstream_steps(self, prompt, start_node_id, debug=False):
        return 2

    def _encode_with_break_isolation(self, clip, prompt_text, normalization, is_sdxl, debug):
        seq_len = 2 if prompt_text == "short" else 5
        cond = torch.ones((1, seq_len, 3))
        pooled = torch.ones((1, 3))
        return cond, pooled


class TestPromptNodeCache(unittest.TestCase):
    def test_scheduled_padding_does_not_mutate_encoded_cache(self):
        node = CacheProbePromptNode()
        clip = FakeClip()

        node.encode(clip, "[short:long:0.5]", prompt={}, unique_id="1")

        cached_shapes = {
            key[2]: cond.shape[1]
            for key, (cond, _pooled) in node._encoded_cache.items()
        }
        self.assertEqual(cached_shapes["short"], 2)
        self.assertEqual(cached_shapes["long"], 5)


if __name__ == "__main__":
    unittest.main()
