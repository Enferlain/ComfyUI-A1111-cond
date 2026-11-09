"""Production mode selection, cache separation and scoped attention behavior."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F
import comfy_stubs
from A1111_Prompt_Node.encoding import get_encoder
from A1111_Prompt_Node.encoding.anima_attention import AnimaAttention, attention_hooks
from A1111_Prompt_Node.encoding.anima_tokenization import join_weighted_segments, weights_for_offsets
from A1111_Prompt_Node.nodes.prompt_node import A1111PromptNode
from A1111_Prompt_Node.nodes.negative_node import A1111PromptNegative
from test_encoder_metadata import AnimaClip, Clip


class Anima:
    pass


class Executor:
    class_obj = Anima()

    def __init__(self, forward):
        self.forward = forward

    def __call__(self, *args, **kwargs):
        return self.forward(kwargs["transformer_options"])


def attention(q, k, v, heads, mask=None, skip_reshape=False, **kwargs):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


class TestAnimaModes(unittest.TestCase):
    def test_optional_choices_and_defaults_on_both_nodes(self):
        for node in (A1111PromptNode, A1111PromptNegative):
            choices, settings = node.INPUT_TYPES()["optional"]["anima_emphasis"]
            self.assertIsInstance(choices, list)
            self.assertEqual(choices, ["native", "single_pass", "attention"])
            self.assertEqual(settings["default"], "native")

    def test_other_encoders_reject_non_native_choices(self):
        for choice in ("single_pass", "attention", "typo"):
            with self.assertRaises(ValueError):
                get_encoder(Clip(), choice)
        self.assertEqual(get_encoder(AnimaClip()).emphasis, "native")

    def test_modes_do_not_share_cached_encodings(self):
        clip, node = AnimaClip(), A1111PromptNode()
        clip.tokenizer = object()
        tokens = clip.tokenize("(cat:1.3)")
        # Stub only the real-tokenizer boundary; encode and cache are production.
        with patch("A1111_Prompt_Node.encoding.anima_tokenization.single_pass_tokens", return_value=(tokens, {})):
            outputs = {}
            for mode in ("native", "single_pass", "attention", "native", "attention"):
                outputs[mode] = node.encode(clip, "(cat:1.3)", anima_emphasis=mode)["result"][0][0][1]
        self.assertEqual(len(clip.calls), 3)
        self.assertEqual(len(node._encoded_cache), 3)
        self.assertNotIn("hooks", outputs["native"])
        self.assertNotIn("hooks", outputs["single_pass"])
        self.assertIn("hooks", outputs["attention"])
        self.assertTrue(torch.all(outputs["attention"]["t5xxl_weights"] == 1))
        self.assertTrue(torch.any(outputs["single_pass"]["t5xxl_weights"] != 1))

    def test_negative_gets_its_own_scoped_hook(self):
        clip = AnimaClip()
        clip.tokenizer = object()
        tokens = clip.tokenize("(cat:1.3)")
        with patch("A1111_Prompt_Node.encoding.anima_tokenization.single_pass_tokens", return_value=(tokens, {})):
            metadata = A1111PromptNegative().encode(clip, "(cat:1.3)", anima_emphasis="attention")["result"][0][0][1]
        self.assertEqual(metadata["hooks"].hook.hook_scope, comfy_stubs.mock_hooks.EnumHookScope.HookedOnly)
        self.assertTrue(torch.all(metadata["t5xxl_weights"] == 1))

    def test_neutral_attention_needs_no_hook_and_bad_weights_fail(self):
        self.assertIsNone(attention_hooks([1, 1]))
        for weights in ([0], [-1], [float("nan")], [float("inf")], [[1]]):
            with self.assertRaises(ValueError):
                attention_hooks(weights)

    def test_negative_preserves_first_step_schedule_fallback(self):
        for mode in ("single_pass", "attention"):
            clip = AnimaClip()
            clip.tokenizer = object()
            with patch("A1111_Prompt_Node.encoding.anima_tokenization.single_pass_tokens", return_value=(clip.tokenize("cat"), {})) as tokenize:
                A1111PromptNegative().encode(clip, "[cat:dog:0.5]", anima_emphasis=mode)
            tokenize.assert_called_once_with(clip.tokenizer, "cat")

    def test_production_span_mapping_keeps_special_tokens_neutral(self):
        text = join_weighted_segments([("@猫", 3), (" ", 1), ("art", .5), ("ist", 1)])
        weights, crossings = weights_for_offsets(text, [(0, 0), (0, 1), (1, 2), (2, 9)])
        self.assertEqual(weights, [1, 3, 3, .75])
        self.assertEqual(crossings, [3])
        with self.assertRaises(ValueError):
            weights_for_offsets(text, [(0, 100)])

    def test_all_modes_keep_anima_capability_limits(self):
        for mode in ("native", "single_pass", "attention"):
            for node in (A1111PromptNode(), A1111PromptNegative()):
                for text, kwargs in (("cat BREAK dog", {}), ("cat", {"normalization": True})):
                    with self.subTest(mode=mode, node=type(node).__name__, text=text):
                        with self.assertRaises(ValueError):
                            node.encode(AnimaClip(), text, anima_emphasis=mode, **kwargs)
            with self.assertRaisesRegex(ValueError, "scheduling/alternation"):
                A1111PromptNode().encode(AnimaClip(), "[cat:dog:0.5]", anima_emphasis=mode)


class TestScopedAnimaAttention(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(sys.modules, {"comfy.ldm.anima.model": SimpleNamespace(Anima=Anima)}))
        self.q = torch.zeros(2, 1, 1, 1)
        self.k = torch.zeros(2, 1, 512, 1)
        self.v = self.k.clone()
        self.v[:, :, 0] = 1

    def call(self, options, kind="attn2_patch", **kwargs):
        for marker in options["patches"][kind]:
            marker(self.q, self.k, self.v, extra_options=options)
        return options["optimized_attention_override"](attention, self.q, self.k, self.v, 1, skip_reshape=True, **kwargs)

    def test_self_attention_with_same_context_length_is_not_biased(self):
        def forward(options):
            self_out = self.call(options, "attn1_patch")
            cross_out = self.call(options)
            return self_out, cross_out
        plain, weighted = AnimaAttention([2])(Executor(forward), transformer_options={})
        self.assertTrue(torch.allclose(plain, torch.full_like(plain, 1 / 512)))
        self.assertTrue(torch.allclose(weighted, torch.full_like(weighted, 2 / 513)))

    def test_deemphasis_masks_and_previous_override_are_composed(self):
        previous = Mock(side_effect=lambda original, *args, **kwargs: original(*args, **kwargs))
        for mask in (torch.tensor([[True] * 511 + [False]]), torch.tensor([[0.] * 511 + [float("-inf")]], dtype=torch.float64)):
            output = AnimaAttention([.5])(Executor(lambda opts: self.call(opts, mask=mask)),
                                          transformer_options={"optimized_attention_override": previous})
            self.assertTrue(torch.allclose(output, torch.full_like(output, .5 / 510.5)))
        self.assertEqual(previous.call_count, 2)

    def test_options_not_mutated_and_repeated_calls_do_not_accumulate(self):
        original = {"patches": {"unrelated": [object()]}}
        wrapper = AnimaAttention([3])
        first = wrapper(Executor(self.call), transformer_options=original)
        second = wrapper(Executor(self.call), transformer_options=original)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(set(original), {"patches"})
        self.assertEqual(set(original["patches"]), {"unrelated"})

    def test_missing_patch_support_and_non_anima_fail(self):
        wrapper = AnimaAttention([2])
        with self.assertRaisesRegex(ValueError, "does not support"):
            wrapper(Executor(lambda options: None), transformer_options={})
        executor = Executor(self.call)
        executor.class_obj = object()
        with self.assertRaisesRegex(ValueError, "Anima diffusion"):
            wrapper(executor, transformer_options={})

    def test_repeated_or_missing_cross_calls_fail_closed(self):
        wrapper = AnimaAttention([2])

        def repeated(options):
            self.call(options)
            return options["optimized_attention_override"](attention, self.q, self.k, self.v, 1, skip_reshape=True)

        with self.assertRaisesRegex(ValueError, "repeated"):
            wrapper(Executor(repeated), transformer_options={})

        def missing(options):
            self.call(options)
            for marker in options["patches"]["attn2_patch"]:
                marker(self.q, self.k, self.v, extra_options=options)

        with self.assertRaisesRegex(ValueError, "patch/call mismatch"):
            wrapper(Executor(missing), transformer_options={})

    def test_context_mismatch_and_nested_hooks_fail(self):
        wrapper = AnimaAttention([2])
        with self.assertRaisesRegex(ValueError, "multiple"):
            wrapper(Executor(self.call), transformer_options={"a1111_anima_attention": True})
        self.k = self.k[:, :, :511]
        self.v = self.v[:, :, :511]
        with self.assertRaisesRegex(ValueError, "context length"):
            wrapper(Executor(self.call), transformer_options={})


if __name__ == "__main__":
    unittest.main()
