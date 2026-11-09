"""Experimental single-pass Anima tokenization; not used by production nodes.

Reuse ComfyUI's emphasis parser, reconstruct the complete text, then tokenize
each stream once. T5 fast-tokenizer character offsets attach weights without
fragmenting either token stream. Tokens crossing weight boundaries receive the
mean weight of their non-whitespace characters; special tokens stay neutral.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class WeightedText:
    text: str
    character_weights: tuple


def join_weighted_segments(segments):
    text = []
    weights = []
    for segment, weight in segments:
        weight = float(weight)
        if not math.isfinite(weight):
            raise ValueError("Emphasis weights must be finite.")
        text.append(segment)
        weights.extend([weight] * len(segment))
    return WeightedText("".join(text), tuple(weights))


def parse_emphasis(text):
    from comfy.sd1_clip import escape_important, token_weights, unescape_important

    if "embedding:" in text:
        raise ValueError("This experiment does not support embedding expansion.")
    segments = token_weights(escape_important(text), 1.0)
    return join_weighted_segments(
        (unescape_important(segment), weight) for segment, weight in segments
    )


def weights_for_offsets(weighted_text, offsets):
    weights = []
    crossings = []
    for index, (start, end) in enumerate(offsets):
        if not 0 <= start <= end <= len(weighted_text.text):
            raise ValueError("Tokenizer offset is outside the reconstructed text.")
        values = [
            weighted_text.character_weights[pos]
            for pos in range(start, end)
            if not weighted_text.text[pos].isspace()
        ]
        weights.append(sum(values) / len(values) if values else 1.0)
        if len(set(values)) > 1:
            crossings.append(index)
    return weights, crossings


def single_pass_tokens(tokenizer, text):
    weighted_text = parse_emphasis(text)
    qwen = tokenizer.qwen3_06b.tokenize_with_weights(
        weighted_text.text, disable_weights=True
    )
    t5 = tokenizer.t5xxl.tokenize_with_weights(
        weighted_text.text, disable_weights=True
    )
    offsets = tokenizer.t5xxl.tokenizer(
        weighted_text.text, return_offsets_mapping=True
    )
    if len(t5) != 1 or [item[0] for item in t5[0]] != offsets["input_ids"]:
        raise ValueError("T5 offset IDs do not match ComfyUI's full-text IDs.")
    weights, crossings = weights_for_offsets(weighted_text, offsets["offset_mapping"])
    tokens = {
        "qwen3_06b": qwen,
        "t5xxl": [[(item[0], weight) for item, weight in zip(t5[0], weights)]],
    }
    return tokens, {
        "clean_text": weighted_text.text,
        "t5_offsets": list(offsets["offset_mapping"]),
        "boundary_crossings": crossings,
    }
