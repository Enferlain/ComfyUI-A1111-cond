"""
A1111 Style Negative Prompt Node

Uses the shared encoding policy without attaching a sampling schedule hook.

If scheduling/alternation syntax is detected, it uses the first step's
prompt and logs an info message.
"""

import logging
from ..parser import expand_wildcards, get_prompt_schedule
from ..metadata import record_prompt_metadata
from ..encoding import get_encoder, ANIMA_EMPHASIS_MODES

logger = logging.getLogger("A1111PromptNode")


class A1111PromptNegative:
    """
    A1111 Style Negative Prompt Node.

    Uses the same encoder/model contract as the positive node.

    If scheduling/alternation syntax is detected, it uses the first step's
    prompt and logs an info message.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": {
                "normalization": (
                    "BOOLEAN",
                    {"default": False, "label_on": "Enable", "label_off": "Disable"},
                ),
                "debug": (
                    "BOOLEAN",
                    {"default": False, "label_on": "Enable", "label_off": "Disable"},
                ),
                "anima_emphasis": (list(ANIMA_EMPHASIS_MODES), {"default": "native", "tooltip": "Anima only: native weighting, stable single-pass weighting, or attention bias. Applies to this negative prompt only."}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "conditioning/advanced"
    OUTPUT_NODE = True

    def encode(
        self,
        clip,
        text,
        normalization=False,
        debug=False,
        unique_id=None,
        extra_pnginfo=None,
        anima_emphasis="native",
    ):
        """
        Encode negative prompt. Uses first step only if scheduling is detected.
        """
        if clip is None:
            raise RuntimeError("ERROR: clip input is None")

        expanded_text = expand_wildcards(text)
        encoder = get_encoder(clip, anima_emphasis)
        encoder.validate_request(expanded_text, normalization=normalization)
        record_prompt_metadata(
            extra_pnginfo,
            unique_id,
            "A1111PromptNegative",
            text,
            expanded_text,
        )

        # Parse the prompt schedule (use dummy steps as it's not supported for negative)
        schedule = get_prompt_schedule(expanded_text, steps=20)

        # Check if scheduling/alternation was used
        if len(schedule) > 1:
            logger.info(
                "[A1111 Prompt Negative] Scheduling/alternation detected but not supported "
                "for negative conditioning. Using first step's prompt only."
            )

        # Always use the first step's prompt
        prompt_text = schedule[0][1] if schedule else ""

        if debug:
            logger.info("[A1111 Prompt Negative] ========== DEBUG START ==========")
            logger.info(
                f"[A1111 Prompt Negative] Input text: {text[:100]}..."
                if len(text) > 100
                else f"[A1111 Prompt Negative] Input text: {text}"
            )
            if expanded_text != text:
                logger.info(
                    f"[A1111 Prompt Negative] Effective text: {expanded_text[:100]}..."
                    if len(expanded_text) > 100
                    else f"[A1111 Prompt Negative] Effective text: {expanded_text}"
                )
            logger.info(
                f"[A1111 Prompt Negative] Encoder: {encoder.name}"
            )
            if len(schedule) > 1:
                logger.info(
                    f"[A1111 Prompt Negative] Schedule had {len(schedule)} segments, using first only"
                )
            logger.info(
                f"[A1111 Prompt Negative] Resolved prompt: {prompt_text[:100]}..."
            )

        cond, metadata = encoder.encode(clip, prompt_text, normalization, debug)

        return {
            "ui": {"text": [expanded_text]},
            "result": ([[cond, metadata]],),
        }
