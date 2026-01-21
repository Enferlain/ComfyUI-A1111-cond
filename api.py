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
                # Word-by-word tokenization with position tracking
                # Split into words while preserving positions
                word_pattern = r"(\S+)"
                words_with_pos = []

                for match in re.finditer(word_pattern, segment):
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
                    # Tokenize this word (without special tokens)
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
