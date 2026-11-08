"""Import API helpers without ComfyUI, and verify live route registration."""

import asyncio
import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeRoutes:
    def __init__(self):
        self.handlers = {}

    def register(self, method, path):
        def decorate(handler):
            self.handlers[method, path] = handler
            return handler

        return decorate

    def post(self, path):
        return self.register("POST", path)

    def get(self, path):
        return self.register("GET", path)


class TestApiImports(unittest.TestCase):
    def load_api(self, prompt_server=None, missing=()):
        self.enterContext(patch.dict(sys.modules))
        for name in list(sys.modules):
            if name == "api" or name.startswith("api."):
                del sys.modules[name]

        aiohttp = types.ModuleType("aiohttp")
        aiohttp.web = types.SimpleNamespace(
            json_response=lambda data, status=200: (status, data)
        )
        server = types.ModuleType("server")
        server.PromptServer = prompt_server
        sys.modules.update(aiohttp=aiohttp, server=server)
        for name in missing:
            sys.modules[name] = None

        # A real server import should register routes, without scanning user files.
        warmup = self.enterContext(patch("threading.Thread.start"))
        api = importlib.import_module("api")
        tokenize = importlib.import_module("api.tokenize")
        autocomplete = importlib.import_module("api.autocomplete")
        return api, tokenize, autocomplete, warmup

    def assert_helpers_available(self, api, tokenize, autocomplete):
        self.assertEqual(tokenize.strip_a1111_syntax("(cat:1.3)"), "cat")
        self.assertIs(api.TagDatabase, autocomplete.TagDatabase)
        self.assertIs(api.get_tokenizer, tokenize.get_tokenizer)

    def test_missing_server_dependency(self):
        api, tokenize, autocomplete, warmup = self.load_api(missing=("server",))
        self.assert_helpers_available(api, tokenize, autocomplete)
        warmup.assert_not_called()

    def test_missing_aiohttp_dependency(self):
        api, tokenize, autocomplete, warmup = self.load_api(missing=("aiohttp",))
        self.assert_helpers_available(api, tokenize, autocomplete)
        warmup.assert_not_called()

    def test_prompt_server_not_created(self):
        class PromptServer:
            pass

        api, tokenize, autocomplete, warmup = self.load_api(PromptServer)
        self.assert_helpers_available(api, tokenize, autocomplete)
        warmup.assert_not_called()

    def test_prompt_server_instance_is_none(self):
        class PromptServer:
            instance = None

        api, tokenize, autocomplete, warmup = self.load_api(PromptServer)
        self.assert_helpers_available(api, tokenize, autocomplete)
        warmup.assert_not_called()

    def test_live_server_registers_all_routes_and_serves_token_counts(self):
        routes = FakeRoutes()
        prompt_server = types.SimpleNamespace(
            instance=types.SimpleNamespace(routes=routes)
        )
        api, tokenize, autocomplete, warmup = self.load_api(prompt_server)
        self.assert_helpers_available(api, tokenize, autocomplete)
        warmup.assert_called_once()
        self.assertEqual(
            set(routes.handlers),
            {
                ("POST", "/a1111_prompt/tokenize"),
                ("POST", "/a1111_prompt/autocomplete"),
                ("POST", "/a1111_prompt/autocomplete/warmup"),
                ("GET", "/a1111_prompt/autocomplete/wildcards"),
                ("GET", "/a1111_prompt/autocomplete/status"),
                ("GET", "/a1111_prompt/autocomplete/files"),
            },
        )

        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return [1]

            def decode(self, ids):
                return "cat"

        class Request:
            async def json(self):
                return {"text": "cat"}

        tokenize._tokenizer = types.SimpleNamespace(tokenizer=Tokenizer())
        handler = routes.handlers["POST", "/a1111_prompt/tokenize"]
        status, data = asyncio.run(handler(Request()))
        self.assertEqual(status, 200)
        self.assertEqual(data["sequences"], [1])


if __name__ == "__main__":
    unittest.main()
