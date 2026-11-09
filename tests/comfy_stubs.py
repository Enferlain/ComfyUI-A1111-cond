"""Minimal ComfyUI dependency setup for isolated node/encoder unit tests."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

mock_model_management = MagicMock()
mock_model_management.intermediate_device.return_value = "cpu"

mock_comfy = MagicMock()
mock_comfy.model_management = mock_model_management

mock_hooks = MagicMock()


class DummyHook:
    def __init__(self, *args, **kwargs):
        self.transformers_dict = kwargs.get("transformers_dict")
        self.hook_scope = kwargs.get("hook_scope")

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
