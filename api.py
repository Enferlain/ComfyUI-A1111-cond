"""
Token Counter API for A1111 Prompt Node

Provides a REST endpoint for live tokenization feedback.
"""

from aiohttp import web
import server

# Global cache for CLIP references (set when node executes)
_clip_cache = {}


def register_clip(clip_id: str, clip):
    """Called from node to register CLIP for tokenization."""
    _clip_cache[clip_id] = clip


def get_cached_clip():
    """Get the most recently registered CLIP."""
    if _clip_cache:
        return list(_clip_cache.values())[-1]
    return None


@server.PromptServer.instance.routes.post("/a1111_prompt/tokenize")
async def tokenize_prompt(request):
    """
    API endpoint for live tokenization using ComfyUI's native CLIP.

    Returns token count per 77-token sequence.
    """
    data = await request.json()
    text = data.get("text", "")

    clip = get_cached_clip()

    if clip is None:
        # Fallback: estimate tokens (~4 chars per token)
        estimated = max(1, len(text) // 4)
        return web.json_response({"sequences": [min(75, estimated)], "estimated": True})

    try:
        tokens = clip.tokenize(text)
        batches = tokens.get("l", [[]])

        # Each batch is 77 tokens (75 usable + start/end)
        # Subtract 2 for start/end tokens
        sequences = [max(0, len(b) - 2) for b in batches]

        return web.json_response(
            {"sequences": sequences if sequences else [0], "estimated": False}
        )
    except Exception as e:
        return web.json_response(
            {"error": str(e), "sequences": [0], "estimated": True}, status=500
        )
