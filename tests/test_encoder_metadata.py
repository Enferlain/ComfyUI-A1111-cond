"""Verify real node outputs with independent Qwen/T5 streams and CLIP weights."""

import re
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

import comfy_stubs  # noqa: F401 - install isolated ComfyUI dependencies
from A1111_Prompt_Node.nodes.prompt_node import A1111PromptNode
from A1111_Prompt_Node.nodes.negative_node import A1111PromptNegative
from A1111_Prompt_Node.encoding import get_encoder


class AnimaClip:
    def __init__(self):
        self.cond_stage_model = SimpleNamespace(clip_name="qwen3_06b")
        self.calls = []

    def tokenize(self, text):
        words = re.sub(r"[():\d.]", "", text).split() or [""]
        qwen = [(i + 10, 1.0) for i in range(len(words))]
        # The T5 stream intentionally has a different length and weighting.
        t5 = [(i + 40, 1.3 if ":1.3" in text and i == 1 else 1.0)
              for i in range(len(words) + 2)]
        return {"qwen3_06b": [qwen], "t5xxl": [t5]}

    def encode_from_tokens(self, tokens, return_dict=False):
        assert return_dict
        assert set(tokens) == {"qwen3_06b", "t5xxl"}
        assert len(tokens["qwen3_06b"]) == len(tokens["t5xxl"]) == 1
        self.calls.append(tokens)
        qwen = tokens["qwen3_06b"][0]
        t5 = tokens["t5xxl"][0]
        return {
            "cond": torch.tensor([[[float(token[0])] * 3 for token in qwen]]),
            "pooled_output": None,
            "attention_mask": torch.ones(1, len(qwen)),
            "t5xxl_ids": torch.tensor([token[0] for token in t5]),
            "t5xxl_weights": torch.tensor([token[1] for token in t5]),
        }


class Clip:
    def __init__(self, sdxl=False):
        self.layer_idx = None
        self.patcher = SimpleNamespace(load_device="cpu")
        self.tokens_seen = None
        self.cond_stage_model = SimpleNamespace(
            reset_clip_options=lambda: None,
            set_clip_options=lambda options: None,
            encode_token_weights=self.encode_token_weights,
        )
        if sdxl:
            self.cond_stage_model.clip_l = object()
            self.cond_stage_model.clip_g = object()

    def load_model(self):
        pass

    def tokenize(self, text):
        return {"l": [[(1, 1.3), (2, 1.0)]]}

    def encode_token_weights(self, tokens):
        self.tokens_seen = tokens
        return torch.ones(1, 2, 3), None, {"attention_mask": torch.ones(1, 2)}


class XLClip(Clip):
    def __init__(self):
        super().__init__(sdxl=True)

    def tokenize(self, text):
        return {"l": [[(1, 1.3), (2, 1.0)]], "g": [[(3, 0.7), (4, 1.0)]]}

    def encode_token_weights(self, tokens):
        self.tokens_seen = tokens
        return torch.ones(1, 2, 2048), torch.full((1, 1280), 2.0)


class ChunkedClip(Clip):
    def tokenize(self, text):
        token_id = {"left": 11, "right": 22, "": 0}[text]
        return {"l": [[(token_id, 1.0)] * 77]}

    def encode_token_weights(self, tokens):
        self.tokens_seen = tokens
        flat = [token_id for batch in tokens["l"] for token_id, _ in batch]
        return torch.tensor(flat, dtype=torch.float32).reshape(1, -1, 1), None


class TestEncoderMetadata(unittest.TestCase):
    def test_selection_declares_family_capabilities(self):
        clip = get_encoder(Clip())
        xl = get_encoder(XLClip())
        anima = get_encoder(AnimaClip())
        self.assertEqual(clip.name, "SD1.5 CLIP")
        self.assertEqual(xl.name, "SDXL CLIP-L/G")
        self.assertTrue(clip.capabilities.breaks)
        self.assertTrue(xl.capabilities.scheduling)
        self.assertFalse(anima.capabilities.breaks)
        self.assertFalse(anima.capabilities.scheduling)
        self.assertFalse(anima.capabilities.normalization)

    def test_unknown_and_mixed_streams_do_not_fall_back_to_clip(self):
        for streams in ({"t5xxl": []}, {"l": [], "g": [], "t5xxl": []}):
            clip = Clip()
            clip.tokenize = lambda text: streams
            with self.subTest(streams=streams):
                for node in (A1111PromptNode(), A1111PromptNegative()):
                    with self.assertRaisesRegex(ValueError, "Unsupported text encoder"):
                        node.encode(clip, "cat")
                self.assertIsNone(clip.tokens_seen)

    def test_anima_selection_requires_its_consuming_model_contract(self):
        clip = AnimaClip()
        clip.cond_stage_model.clip_name = "another_qwen_model"
        with self.assertRaisesRegex(ValueError, "Unsupported text encoder"):
            get_encoder(clip)

    def test_missing_consuming_model_contract_reports_unsupported_encoder(self):
        clip = AnimaClip()
        del clip.cond_stage_model
        with self.assertRaisesRegex(ValueError, "Unsupported text encoder"):
            get_encoder(clip)

    def test_clip_break_preserves_separate_full_batches_for_both_nodes(self):
        for node in (A1111PromptNode(), A1111PromptNegative()):
            clip = ChunkedClip()
            cond, _ = node.encode(clip, "left BREAK right")["result"][0][0]
            self.assertEqual([len(batch) for batch in clip.tokens_seen["l"]], [77, 77])
            self.assertTrue(torch.equal(cond[0, :77, 0], torch.full((77,), 11.0)))
            self.assertTrue(torch.equal(cond[0, 77:, 0], torch.full((77,), 22.0)))

    def test_clip_empty_break_preserves_empty_prompt_fallback(self):
        clip = ChunkedClip()
        cond, _ = A1111PromptNode().encode(clip, " BREAK ")["result"][0][0]
        self.assertEqual(cond.shape[1], 77)
        self.assertTrue(torch.equal(cond, torch.zeros_like(cond)))

    def test_encoder_rechecks_streams_for_actual_prompt(self):
        clip = Clip()
        encoder = get_encoder(clip)
        clip.tokenize = lambda text: {"t5xxl": []}
        with self.assertRaisesRegex(ValueError, "unexpected token streams"):
            encoder.encode(clip, "cat")

    def test_anima_static_positive_matches_core_including_extras(self):
        clip = AnimaClip()
        expected = clip.encode_from_tokens(clip.tokenize("a cat sitting"), return_dict=True)
        cond, metadata = A1111PromptNode().encode(clip, "a cat sitting")["result"][0][0]
        self.assertTrue(torch.equal(cond, expected["cond"]))
        self.assertEqual(set(metadata), set(expected) - {"cond"})
        for key in ("attention_mask", "t5xxl_ids", "t5xxl_weights"):
            self.assertTrue(torch.equal(metadata[key], expected[key]), key)

    def test_anima_negative_preserves_native_emphasis_channel(self):
        clip = AnimaClip()
        text = "a (cat:1.3) sitting"
        expected = clip.encode_from_tokens(clip.tokenize(text), return_dict=True)
        cond, metadata = A1111PromptNegative().encode(clip, text)["result"][0][0]
        self.assertTrue(torch.equal(cond, expected["cond"]))
        self.assertTrue(torch.equal(metadata["t5xxl_weights"], expected["t5xxl_weights"]))
        self.assertAlmostEqual(metadata["t5xxl_weights"][1].item(), 1.3)

    def test_anima_break_is_explicitly_unsupported_for_both_nodes(self):
        for node in (A1111PromptNode(), A1111PromptNegative()):
            for text in ("short BREAK much longer text", " BREAK "):
                clip = AnimaClip()
                with self.subTest(node=type(node).__name__, text=text):
                    with self.assertRaisesRegex(ValueError, "Anima BREAK is not supported"):
                        node.encode(clip, text)
                self.assertEqual(clip.calls, [])

    def test_anima_schedule_and_normalization_are_explicitly_unsupported(self):
        for text, normalization, message in (
            ("[cat:dog:0.5]", False, "scheduling/alternation"),
            ("[cat|dog]", False, "scheduling/alternation"),
            ("cat", True, "normalization"),
        ):
            clip = AnimaClip()
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, message):
                A1111PromptNode().encode(
                    clip, text, normalization=normalization,
                    prompt={"2": {"class_type": "KSampler", "inputs": {
                        "positive": ["1", 0], "steps": 4,
                    }}}, unique_id="1",
                )
            self.assertEqual(clip.calls, [])

    def test_cached_metadata_dictionary_is_not_exposed_to_consumers(self):
        node = A1111PromptNode()
        clip = AnimaClip()
        first = node.encode(clip, "cat")["result"][0][0]
        del first[1]["t5xxl_ids"]
        second = node.encode(clip, "cat")["result"][0][0]
        self.assertIn("t5xxl_ids", second[1])
        self.assertEqual(len(clip.calls), 1)

    def test_clip_direct_scaling_and_auxiliary_metadata_are_preserved(self):
        clip = Clip()
        cond, metadata = A1111PromptNode().encode(clip, "cat")["result"][0][0]
        self.assertTrue(torch.allclose(cond[0, 0], torch.full((3,), 1.3)))
        self.assertTrue(torch.equal(cond[0, 1], torch.ones(3)))
        self.assertEqual(clip.tokens_seen["l"], [[(1, 1.0), (2, 1.0)]])
        self.assertTrue(torch.equal(metadata["attention_mask"], torch.ones(1, 2)))

    def test_auxiliary_metadata_cannot_silently_enter_tensor_only_schedule(self):
        node = A1111PromptNode()
        clip = Clip()
        encode = Mock(wraps=clip.cond_stage_model.encode_token_weights)
        clip.cond_stage_model.encode_token_weights = encode
        with self.assertRaisesRegex(ValueError, "auxiliary conditioning fields"):
            node.encode(clip, "[cat:dog:0.5]")
        self.assertEqual(encode.call_count, 1)
        self.assertEqual(node._encoded_cache, {})

    def test_cached_auxiliary_metadata_is_rejected_before_new_encoding(self):
        node = A1111PromptNode()
        clip = Clip()
        node.encode(clip, "cat")
        encode = Mock(wraps=clip.cond_stage_model.encode_token_weights)
        clip.cond_stage_model.encode_token_weights = encode
        with self.assertRaisesRegex(ValueError, "auxiliary conditioning fields"):
            node.encode(clip, "[cat:dog:0.5]")
        encode.assert_not_called()

    def test_rejected_prompts_do_not_write_resolved_prompt_metadata(self):
        for node in (A1111PromptNode(), A1111PromptNegative()):
            info = {}
            with self.assertRaisesRegex(ValueError, "Anima BREAK"):
                node.encode(AnimaClip(), "cat BREAK dog", unique_id="1", extra_pnginfo=info)
            self.assertEqual(info, {})

    def test_sdxl_keeps_separate_l_and_g_scaling_and_pooled_output(self):
        clip = XLClip()
        cond, metadata = A1111PromptNode().encode(clip, "cat")["result"][0][0]
        self.assertTrue(torch.allclose(cond[0, 0, :768], torch.full((768,), 1.3)))
        self.assertTrue(torch.allclose(cond[0, 0, 768:], torch.full((1280,), 0.7)))
        self.assertTrue(torch.equal(cond[0, 1], torch.ones(2048)))
        self.assertTrue(torch.equal(metadata["pooled_output"], torch.full((1, 1280), 2.0)))
        for channel in ("l", "g"):
            self.assertTrue(all(weight == 1.0 for _, weight in clip.tokens_seen[channel][0]))

    def test_other_dual_stream_layouts_are_not_scaled_as_sdxl(self):
        clip = XLClip()
        clip.cond_stage_model.encode_token_weights = lambda tokens: (torch.ones(1, 2, 4096), None)
        with self.assertRaisesRegex(ValueError, "separate encoder implementation"):
            A1111PromptNode().encode(clip, "cat")


if __name__ == "__main__":
    unittest.main()
