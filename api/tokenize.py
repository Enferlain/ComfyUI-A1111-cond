"""
Token Counter API for A1111 Prompt Node

Provides a REST endpoint for live tokenization feedback.
Uses word-by-word tokenization with manual position tracking.
"""

from aiohttp import web
import server
import re

# Lazy-loaded tokenizer instance
_tokenizer = None


def get_tokenizer():
    """Get or create the SD tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        from comfy.sd1_clip import SDTokenizer

        _tokenizer = SDTokenizer()
    return _tokenizer


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

    # Handle bracket expressions: [A|B|C] or [from:to:when]
    # Keep the LONGEST option for worst-case token counting
    def keep_longest_option(match):
        content = match.group(1)

        if "|" in content:
            # It's alternation [A|B|C] - find longest option
            options = content.split("|")
            # Remove any trailing :number from last option (scheduled alternation)
            # e.g., [A|B:0.5] -> options = ["A", "B:0.5"]
            last = options[-1]
            if ":" in last:
                # Check if it ends with :number
                colon_match = re.match(r"^(.+?)::?[\d.]+$", last)
                if colon_match:
                    options[-1] = colon_match.group(1)

            # Return the longest option
            return max(options, key=len)

        elif ":" in content:
            # It's scheduling [from:to:when] - keep longer of from/to
            parts = content.split(":")
            if len(parts) >= 2:
                # parts[0] = from, parts[1] = to, parts[2+] = when (numbers)
                from_part = parts[0]
                to_part = parts[1] if len(parts) > 1 else ""
                # Return the longer of from/to
                return from_part if len(from_part) >= len(to_part) else to_part
            return content

        else:
            # Simple bracket emphasis [word] - just return content
            return content

    # Process bracket expressions - non-greedy match for innermost brackets
    # Repeat to handle nested structures
    prev_text = None
    while prev_text != text:
        prev_text = text
        text = re.sub(r"\[([^\[\]]*)\]", keep_longest_option, text)

    # Remove weight specifications: :1.2) at end of emphasis
    # This handles (word:1.2)
    text = re.sub(r":[\d.]+(?=\))", "", text)

    # Remove any remaining weight specs that might be floating
    text = re.sub(r":[\d.]+(?=\s|$)", "", text)

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


@server.PromptServer.instance.routes.post("/a1111_prompt/tokenize")
async def tokenize_prompt(request):
    """
    API endpoint for live tokenization.

    Returns token count per 77-token sequence and character positions
    where boundaries fall using word-by-word tokenization.
    BREAK forces a new sequence (matching A1111/parser behavior).
    """
    data = await request.json()
    text = data.get("text", "")

    try:
        tokenizer = get_tokenizer()
        hf_tokenizer = tokenizer.tokenizer  # Access underlying HuggingFace tokenizer

        # Split by BREAK first (matching parser.py behavior)
        break_pattern = r"\s*\bBREAK\b\s*"
        break_matches = list(re.finditer(break_pattern, text))
        break_segments = re.split(break_pattern, text)

        sequences = []
        boundaries = []
        tokens_detail = []  # Per-token info: [{text, id, chunk_idx}, ...]

        # Track position in original text
        current_text_offset = 0
        current_chunk_idx = 0

        for seg_idx, segment in enumerate(break_segments):
            # Find where this segment starts in the original text
            if segment:
                segment_start = text.find(segment, current_text_offset)
            else:
                segment_start = current_text_offset

            segment_text = segment.strip()

            if not segment_text:
                sequences.append(0)
                current_chunk_idx += 1
            else:
                # FIRST: Strip A1111 syntax from entire segment
                # This handles bracket expressions with spaces like [A|B C|D]
                clean_segment = strip_a1111_syntax(segment_text)

                if not clean_segment:
                    sequences.append(0)
                    current_chunk_idx += 1
                    current_text_offset = segment_start + len(segment)
                    continue

                # Word-by-word tokenization with position tracking
                # Split into words from the CLEANED segment
                word_pattern = r"(\S+)"
                words_with_pos = []

                for match in re.finditer(word_pattern, clean_segment):
                    word = match.group(1)
                    # Position relative to segment start
                    word_start = match.start()
                    word_end = match.end()
                    words_with_pos.append((word, word_start, word_end))

                # Tokenize each word and track cumulative token count
                chunk_size = 75
                current_chunk_tokens = 0
                chunk_sequences = []

                for word, word_start_rel, word_end_rel in words_with_pos:
                    # Words are already from the cleaned segment, just tokenize directly
                    word_tokens = hf_tokenizer.encode(word, add_special_tokens=False)
                    word_token_count = len(word_tokens)

                    # Check if adding this word would exceed chunk size
                    if (
                        current_chunk_tokens + word_token_count > chunk_size
                        and current_chunk_tokens > 0
                    ):
                        # Save current chunk
                        chunk_sequences.append(current_chunk_tokens)

                        # Add boundary marker at the END of the previous word
                        # (which is the start of this word)
                        boundary_char = segment_start + word_start_rel
                        boundaries.append({"char_pos": boundary_char, "type": "chunk"})

                        current_chunk_tokens = 0
                        current_chunk_idx += 1

                    # Decode each token to get the text representation
                    for token_id in word_tokens:
                        token_text = hf_tokenizer.decode([token_id])
                        tokens_detail.append(
                            {
                                "text": token_text,
                                "id": token_id,
                                "chunk": current_chunk_idx,
                            }
                        )

                    current_chunk_tokens += word_token_count

                # Don't forget the last chunk
                if current_chunk_tokens > 0:
                    chunk_sequences.append(current_chunk_tokens)

                sequences.extend(chunk_sequences if chunk_sequences else [0])
                current_chunk_idx += 1

            # Move past this segment
            current_text_offset = segment_start + len(segment)

            # Add BREAK boundary if not the last segment
            if seg_idx < len(break_matches):
                boundaries.append(
                    {"char_pos": break_matches[seg_idx].start(), "type": "break"}
                )
                # Add BREAK marker to tokens
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

        # Calculate stats
        word_count = len(text.split())
        char_count = len(text)

        return web.json_response(
            {
                "sequences": sequences,
                "boundaries": boundaries,
                "tokens": tokens_detail,
                "stats": {
                    "total_tokens": sum(sequences),
                    "chunks": len(sequences),
                    "words": word_count,
                    "characters": char_count,
                },
            }
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        return web.json_response(
            {"error": str(e), "sequences": None, "boundaries": None}, status=500
        )
