"""
Classic prompt tokenization helpers.

These helpers let the node opt into two ReForge/A1111-style behaviors without
replacing the default Comfy tokenization path:
1. Classic attention parsing (`[]` de-emphasis, escaped brackets, BREAK markers)
2. Classic chunking (comma backtrack instead of pure 75-token rollover)
"""

import re

from comfy import sd1_clip as comfy_sd1_clip


_RE_ATTENTION = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:\s*([+-]?[.\d]+)\s*\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)

_RE_BREAK = re.compile(r"\s*\bBREAK\b\s*", re.S)


def _parse_classic_attention(text):
    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for match in _RE_ATTENTION.finditer(text):
        token_text = match.group(0)
        weight = match.group(1)

        if token_text.startswith("\\"):
            res.append([token_text[1:], 1.0])
        elif token_text == "(":
            round_brackets.append(len(res))
        elif token_text == "[":
            square_brackets.append(len(res))
        elif weight is not None and round_brackets:
            multiply_range(round_brackets.pop(), float(weight))
        elif token_text == ")" and round_brackets:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif token_text == "]" and square_brackets:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = re.split(_RE_BREAK, token_text)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1.0])
                if part:
                    res.append([part, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        return [("", 1.0)]

    merged = []
    for part_text, weight in res:
        if (
            merged
            and merged[-1][1] == weight
            and merged[-1][0] != "BREAK"
            and part_text != "BREAK"
        ):
            merged[-1][0] += part_text
        else:
            merged.append([part_text, weight])

    return [(part_text, weight) for part_text, weight in merged if part_text or weight < 0]


def _parse_default_attention(text):
    escaped = comfy_sd1_clip.escape_important(text)
    parsed = comfy_sd1_clip.token_weights(escaped, 1.0)

    out = []
    for part_text, weight in parsed:
        unescaped = comfy_sd1_clip.unescape_important(part_text)
        pieces = re.split(_RE_BREAK, unescaped)
        for i, piece in enumerate(pieces):
            if i > 0:
                out.append(("BREAK", -1.0))
            if piece:
                out.append((piece, weight))

    if not out:
        return [("", 1.0)]

    return out


def _get_subtokenizers(tokenizer):
    subtokenizers = {}
    for name in dir(tokenizer):
        if not name.startswith("clip_"):
            continue
        value = getattr(tokenizer, name)
        suffix = name.split("clip_", 1)[1]
        subtokenizers[suffix] = value

    if subtokenizers:
        return subtokenizers

    # Fallback for tokenizers that expose a single subtokenizer directly.
    if hasattr(tokenizer, "tokenize_with_weights"):
        return {"l": tokenizer}

    raise RuntimeError("Unsupported tokenizer layout for classic prompt tokenization.")


def _segment_to_tokens(subtokenizer, text, weight):
    embedding_identifier = subtokenizer.embedding_identifier
    split = re.split(r" {0}|\n{0}".format(re.escape(embedding_identifier)), text)
    to_tokenize = [split[0]]
    for i in range(1, len(split)):
        to_tokenize.append("{}{}".format(embedding_identifier, split[i]))

    flat_tokens = []
    for piece in [x for x in to_tokenize if x != ""]:
        if piece.startswith(embedding_identifier) and subtokenizer.embedding_directory is not None:
            embedding_name = piece[len(embedding_identifier) :].strip("\n")
            embed, leftover = subtokenizer._try_get_embedding(embedding_name)
            if embed is not None:
                if len(embed.shape) == 1:
                    flat_tokens.append((embed, weight))
                else:
                    for row in range(embed.shape[0]):
                        flat_tokens.append((embed[row], weight))
                if leftover:
                    piece = leftover
                else:
                    continue

        end = -1 if subtokenizer.tokenizer_adds_end_token else 999999999999
        token_ids = subtokenizer.tokenizer(piece)["input_ids"][
            subtokenizer.tokens_start : end
        ]
        for token in token_ids:
            flat_tokens.append((token, weight))

    return flat_tokens


def _find_last_comma(tokens, comma_token):
    if comma_token is None:
        return -1
    for index in range(len(tokens) - 1, -1, -1):
        token = tokens[index][0]
        if isinstance(token, int) and token == comma_token:
            return index
    return -1


def _pack_tokens(subtokenizer, parsed_segments, use_classic_chunking):
    has_start = subtokenizer.start_token is not None
    has_end = subtokenizer.end_token is not None
    chunk_length = subtokenizer.max_length - int(has_start) - int(has_end)
    comma_token = None
    if use_classic_chunking:
        comma_token = subtokenizer.tokenizer.get_vocab().get(",</w>", None)

    batches = []
    current = []
    last_comma = -1

    def flush(force=False):
        nonlocal current
        nonlocal last_comma
        if not current and not force:
            return

        batch = []
        if has_start:
            batch.append((subtokenizer.start_token, 1.0))
        batch.extend(current)
        if has_end:
            batch.append((subtokenizer.end_token, 1.0))
        while len(batch) < subtokenizer.max_length:
            batch.append((subtokenizer.pad_token, 1.0))
        batches.append(batch)
        current = []
        last_comma = -1

    for part_text, weight in parsed_segments:
        if part_text == "BREAK" and weight < 0:
            flush(force=False)
            continue

        flat_tokens = _segment_to_tokens(subtokenizer, part_text, weight)
        position = 0
        while position < len(flat_tokens):
            if len(current) == chunk_length:
                flush(force=False)

            token_pair = flat_tokens[position]
            token_value = token_pair[0]

            if use_classic_chunking and isinstance(token_value, int):
                if token_value == comma_token:
                    last_comma = len(current)
                elif (
                    len(current) == chunk_length
                    and last_comma != -1
                    and len(current) - last_comma <= 20
                ):
                    break_location = last_comma + 1
                    reloc = current[break_location:]
                    current = current[:break_location]
                    flush(force=False)
                    current = reloc
                    last_comma = _find_last_comma(current, comma_token)
                    continue

            if len(current) == chunk_length:
                flush(force=False)

            current.append(token_pair)
            position += 1

    flush(force=not batches)
    return batches


def tokenize_prompt(clip, text, use_classic_attention=False, use_classic_chunking=False):
    """
    Build token-weight batches matching Comfy's expected tokenizer output shape.
    """
    parsed_segments = (
        _parse_classic_attention(text)
        if use_classic_attention
        else _parse_default_attention(text)
    )

    tokens = {}
    for key, subtokenizer in _get_subtokenizers(clip.tokenizer).items():
        tokens[key] = _pack_tokens(
            subtokenizer, parsed_segments, use_classic_chunking=use_classic_chunking
        )
    return tokens
