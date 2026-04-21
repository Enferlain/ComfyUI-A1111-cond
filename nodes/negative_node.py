"""
A1111 Style Negative Prompt Node

Same encoding logic as A1111PromptNode but WITHOUT model input.
This prevents alternation hooks from being registered on the model,
which would break generation if used for negative conditioning.

If scheduling/alternation syntax is detected, it uses the first step's
prompt and logs an info message.
"""

import logging
from ..parser import expand_wildcards, get_prompt_schedule
from .prompt_node import A1111PromptNode

logger = logging.getLogger("A1111PromptNode")


class A1111PromptNegative:
    """
    A1111 Style Negative Prompt Node.

    Same encoding logic as A1111PromptNode but WITHOUT model input.
    This prevents alternation hooks from being registered on the model,
    which would break generation if used for negative conditioning.

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
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "conditioning/advanced"
    OUTPUT_NODE = True

    def encode(self, clip, text, normalization=False, debug=False):
        """
        Encode negative prompt. Uses first step only if scheduling is detected.
        """
        if clip is None:
            raise RuntimeError("ERROR: clip input is None")

        expanded_text = expand_wildcards(text)

        is_sdxl = hasattr(clip.cond_stage_model, "clip_l") and hasattr(
            clip.cond_stage_model, "clip_g"
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
                f"[A1111 Prompt Negative] Model: {'SDXL' if is_sdxl else 'SD1.5'}"
            )
            if len(schedule) > 1:
                logger.info(
                    f"[A1111 Prompt Negative] Schedule had {len(schedule)} segments, using first only"
                )
            logger.info(
                f"[A1111 Prompt Negative] Resolved prompt: {prompt_text[:100]}..."
            )

        # Reuse the encoding helper from A1111PromptNode
        encoder = A1111PromptNode()
        cond, pooled = encoder._encode_with_break_isolation(
            clip, prompt_text, normalization, is_sdxl, debug
        )

        return {
            "ui": {"text": [expanded_text]},
            "result": ([[cond, {"pooled_output": pooled}]],),
        }
