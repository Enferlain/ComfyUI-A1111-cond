"""
Token Counter API for A1111 Prompt Node

Provides a REST endpoint for live tokenization feedback.
Uses word-by-word tokenization with manual position tracking.
"""

from aiohttp import web
import server
import logging
import re

try:
    from ..parser.wildcards import expand_wildcards_for_token_count
except ImportError:
    from parser.wildcards import expand_wildcards_for_token_count

# Lazy-loaded tokenizer instance
_tokenizer = None
logger = logging.getLogger("A1111PromptNode")


def _coerce_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _find_word_position(source: str, word: str, start: int, end: int, fallback: int):
    found = source.find(word, start, end)
    if found == -1:
        return fallback
    return found


def get_tokenizer():
    """Get or create the SD tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        from comfy.sd1_clip import SDTokenizer

        _tokenizer = SDTokenizer()
    return _tokenizer


def _encode_word(hf_tokenizer, word: str):
    return hf_tokenizer.encode(word, add_special_tokens=False)


def _decode_token(hf_tokenizer, token_id):
    return hf_tokenizer.decode([token_id])


def strip_a1111_syntax(text: str) -> str:
    r"""
    Strip A1111 emphasis/scheduling syntax from text, leaving only tokenizable content.

    This matches what ComfyUI's clip.tokenize() does internally when it parses
    emphasis syntax like (word:1.2) - only the word gets tokenized, not the
    parentheses or weight numbers.

    For alternation/scheduling, keeps the LONGEST option to show worst-case token count.

    Handles:
    - Emphasis: (word:1.2) -> word, (word) -> word
    - Negative emphasis: [word] -> word, [word:0.5] -> word
    - Scheduling: [from:to:when] -> max(from, to) (keeps longer one)
    - Alternation: [A|B|C] -> max(A, B, C) (keeps longest option)
    - Escaped chars: \( \) \[ \] -> ( ) [ ]
    """
    # First, handle escaped characters - replace with placeholders
    text = text.replace("\\(", "\x00LPAREN\x00")
    text = text.replace("\\)", "\x00RPAREN\x00")
    text = text.replace("\\[", "\x00LBRACK\x00")
    text = text.replace("\\]", "\x00RBRACK\x00")

    # Remove weight specifications: :1.2) at end of emphasis
    # This handles (word:1.2)
    text = re.sub(r":[\d.]+(?=\))", "", text)

    # Remove any remaining weight specs that might be floating
    text = re.sub(r":[\d.]+(?=\s|$)", "", text)

    # Handle bracket expressions: [A|B|C] or [from:to:when]
    # Keep the LONGEST option for worst-case token counting
    def keep_longest_option(match):
        content = match.group(1)

        if "|" in content:
            # It's alternation [A|B|C] - find longest option
            options = content.split("|")
            # Remove any trailing :number from last option (scheduled alternation)
            last = options[-1]
            if ":" in last:
                colon_match = re.match(r"^(.+?)::?[\d.]+$", last)
                if colon_match:
                    options[-1] = colon_match.group(1)

            # Return the longest option
            return max(options, key=len)

        elif ":" in content:
            # It's scheduling [from:to:when] - keep longer of from/to
            parts = content.split(":")
            if len(parts) >= 2:
                from_part = parts[0]
                to_part = parts[1] if len(parts) > 1 else ""

                # Compare based on tokenizable length (ignoring parens) for better worst-case estimation
                clean_from = from_part.replace("(", "").replace(")", "").strip()
                clean_to = to_part.replace("(", "").replace(")", "").strip()

                return from_part if len(clean_from) >= len(clean_to) else to_part
            return content

        else:
            return content

    # Process bracket expressions - non-greedy match for innermost brackets
    prev_text = None
    while prev_text != text:
        prev_text = text
        text = re.sub(r"\[([^\[\]]*)\]", keep_longest_option, text)

    # Remove parentheses (they're just syntax markers now)
    text = text.replace("(", " ")
    text = text.replace(")", " ")

    # Restore escaped characters as literal chars
    text = text.replace("\x00LPAREN\x00", "(")
    text = text.replace("\x00RPAREN\x00", ")")
    text = text.replace("\x00LBRACK\x00", "[")
    text = text.replace("\x00RBRACK\x00", "]")

    # Clean up multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def build_token_info(text: str, counting_text: str, hf_tokenizer):
    is_estimated = counting_text != text

    # Split by BREAK first (matching parser.py behavior)
    break_pattern = r"\s*\bBREAK\b\s*"
    break_matches = list(re.finditer(break_pattern, counting_text))
    break_segments = re.split(break_pattern, counting_text)

    sequences = []
    boundaries = []
    tokens_detail = []  # Per-token info: [{text, id, chunk_idx}, ...]

    # Track position in token-counting text and display/original text separately.
    current_text_offset = 0
    display_search_start = 0
    current_chunk_idx = 0

    for seg_idx, segment in enumerate(break_segments):
        # Find where this segment starts in the counting text.
        if segment:
            segment_start = counting_text.find(segment, current_text_offset)
        else:
            segment_start = current_text_offset

        leading_trim = len(segment) - len(segment.lstrip())
        segment_text_start = segment_start + leading_trim
        segment_text = segment.strip()

        if not segment_text:
            sequences.append(0)
            current_chunk_idx += 1
        else:
            clean_segment = strip_a1111_syntax(segment_text)

            if not clean_segment:
                sequences.append(0)
                current_chunk_idx += 1
                current_text_offset = segment_start + len(segment)
                continue

            word_pattern = r"(\S+)"
            words_with_pos = []

            for match in re.finditer(word_pattern, clean_segment):
                word = match.group(1)
                word_start = match.start()
                word_end = match.end()
                words_with_pos.append((word, word_start, word_end))

            chunk_size = 75
            current_chunk_tokens = 0
            chunk_sequences = []
            counting_search_start = segment_text_start
            segment_end = segment_start + len(segment)

            for word, word_start_rel, _word_end_rel in words_with_pos:
                word_tokens = _encode_word(hf_tokenizer, word)
                word_token_count = len(word_tokens)

                if is_estimated:
                    found = text.find(word, display_search_start)
                    if found != -1:
                        original_word_start = found
                        display_search_start = found + len(word)
                    else:
                        original_word_start = min(display_search_start, len(text))
                else:
                    # Map the cleaned word back into the original segment. This keeps
                    # boundary markers aligned after stripping A1111 syntax.
                    fallback_pos = segment_text_start + word_start_rel
                    original_word_start = _find_word_position(
                        counting_text,
                        word,
                        counting_search_start,
                        segment_end,
                        fallback_pos,
                    )
                    counting_search_start = max(
                        counting_search_start, original_word_start + len(word)
                    )

                if (
                    current_chunk_tokens + word_token_count > chunk_size
                    and current_chunk_tokens > 0
                ):
                    chunk_sequences.append(current_chunk_tokens)
                    boundaries.append(
                        {
                            "char_pos": original_word_start,
                            "type": "chunk",
                            "from_chunk": current_chunk_idx,
                            "to_chunk": current_chunk_idx + 1,
                            "estimated": is_estimated,
                        }
                    )

                    current_chunk_tokens = 0
                    current_chunk_idx += 1

                for token_id in word_tokens:
                    tokens_detail.append(
                        {
                            "text": _decode_token(hf_tokenizer, token_id),
                            "id": token_id,
                            "chunk": current_chunk_idx,
                        }
                    )

                current_chunk_tokens += word_token_count

            if current_chunk_tokens > 0:
                chunk_sequences.append(current_chunk_tokens)

            sequences.extend(chunk_sequences if chunk_sequences else [0])
            current_chunk_idx += 1

        current_text_offset = segment_start + len(segment)

        if seg_idx < len(break_matches):
            break_char_pos = break_matches[seg_idx].start()
            if is_estimated:
                break_char_pos = min(display_search_start, len(text))
            boundaries.append(
                {
                    "char_pos": break_char_pos,
                    "type": "break",
                    "from_chunk": current_chunk_idx - 1,
                    "to_chunk": current_chunk_idx,
                    "estimated": is_estimated,
                }
            )
            tokens_detail.append(
                {
                    "text": "BREAK",
                    "id": None,
                    "chunk": current_chunk_idx - 1,
                    "is_break": True,
                }
            )

    if not sequences:
        sequences = [0]

    return {
        "sequences": sequences,
        "boundaries": boundaries,
        "tokens": tokens_detail,
        "stats": {
            "total_tokens": sum(sequences),
            "chunks": len(sequences),
            "words": len(text.split()),
            "characters": len(text),
            "estimated_from_wildcards": is_estimated,
        },
        "estimated_text": counting_text if is_estimated else None,
    }


@server.PromptServer.instance.routes.post("/a1111_prompt/tokenize")
async def tokenize_prompt(request):
    """
    API endpoint for live tokenization.

    Returns token count per 77-token sequence and character positions
    where boundaries fall using word-by-word tokenization.
    BREAK forces a new sequence (matching A1111/parser behavior).
    """
    try:
        try:
            data = await request.json()
        except Exception:
            return web.json_response(
                {"error": "Invalid JSON", "sequences": None, "boundaries": None},
                status=400,
            )

        if not isinstance(data, dict):
            return web.json_response(
                {
                    "error": "Request body must be a JSON object",
                    "sequences": None,
                    "boundaries": None,
                },
                status=400,
            )

        text = _coerce_text(data.get("text", ""))

        tokenizer = get_tokenizer()
        hf_tokenizer = tokenizer.tokenizer  # Access underlying HuggingFace tokenizer

        def score_token_count(candidate: str) -> int:
            clean_candidate = strip_a1111_syntax(candidate)
            if not clean_candidate:
                return 0
            return len(hf_tokenizer.encode(clean_candidate, add_special_tokens=False))

        try:
            counting_text = expand_wildcards_for_token_count(
                text,
                scorer=score_token_count,
            )
        except Exception as exc:
            logger.warning(
                "[A1111 Prompt] Wildcard token estimate failed; falling back to literal token count: %s",
                exc,
            )
            counting_text = text
        is_estimated = counting_text != text

        return web.json_response(build_token_info(text, counting_text, hf_tokenizer))
    except Exception as e:
        import traceback

        traceback.print_exc()
        return web.json_response(
            {"error": str(e), "sequences": None, "boundaries": None}, status=500
        )
