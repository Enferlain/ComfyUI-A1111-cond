"""
Token Counter API for A1111 Prompt Node

Provides a REST endpoint for live tokenization feedback.
Uses ComfyUI's built-in SDTokenizer directly (no model execution needed).
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

    Returns token count per 77-token sequence.
    BREAK forces a new sequence (matching A1111/parser behavior).
    """
    data = await request.json()
    text = data.get("text", "")

    try:
        tokenizer = get_tokenizer()

        # Split by BREAK first (matching parser.py behavior)
        break_segments = re.split(r"\s*\bBREAK\b\s*", text)

        sequences = []
        for segment in break_segments:
            segment = segment.strip()
            if not segment:
                # Empty segment after BREAK = empty sequence
                sequences.append(0)
                continue

            # Tokenize this segment
            batches = tokenizer.tokenize_with_weights(segment, return_word_ids=True)

            for batch in batches:
                # Count only content tokens (word_id > 0)
                content_tokens = sum(1 for t, w, word_id in batch if word_id > 0)
                sequences.append(content_tokens)

        if not sequences:
            sequences = [0]

        return web.json_response({"sequences": sequences})
    except Exception as e:
        return web.json_response({"error": str(e), "sequences": None}, status=500)
