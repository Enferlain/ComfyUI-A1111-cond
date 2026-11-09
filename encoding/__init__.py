"""Explicit selection of supported encoder/model conditioning contracts."""

from .anima import AnimaEncoder, ANIMA_EMPHASIS_MODES
from .clip import CLIPEncoder

_ENCODERS = (AnimaEncoder(), CLIPEncoder(("l",)), CLIPEncoder(("l", "g")))


def get_encoder(clip, anima_emphasis="native"):
    """Inspect token streams without loading encoder weights; never guess CLIP."""
    if clip is None:
        raise ValueError("clip input is None")
    if anima_emphasis not in ANIMA_EMPHASIS_MODES:
        raise ValueError(f"Unknown Anima emphasis mode: {anima_emphasis}")
    tokens = clip.tokenize("")
    streams = set(tokens) if isinstance(tokens, dict) else set()
    for encoder in _ENCODERS:
        if encoder.matches(clip, streams):
            if isinstance(encoder, AnimaEncoder):
                return encoder if anima_emphasis == "native" else AnimaEncoder(anima_emphasis)
            if anima_emphasis != "native":
                raise ValueError("Anima emphasis modes apply only to Anima. Select native for CLIP/SDXL.")
            return encoder
    raise ValueError(
        f"Unsupported text encoder token streams: {sorted(streams)}. "
        "Supported conditioning paths: SD1.5 CLIP, SDXL CLIP-L/G, and static Anima."
    )


__all__ = ["get_encoder", "ANIMA_EMPHASIS_MODES"]
