"""
A1111 Style Prompt Node (God Node)

Implements A1111-style prompt handling:
- Hard chunking with BREAK isolation (no bleed between segments)
- Direct scaling (no burn) instead of interpolation
- BREAK support with proper token isolation
- Scheduling [from:to:when] (Switch, Add, Remove)
- Alternation [A|B] (true per-step switching via schedule)
"""

import torch
import logging
import re
import comfy.model_management as model_management
from .parser import get_prompt_schedule

logger = logging.getLogger("A1111PromptNode")


class A1111PromptNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": {
                "steps": (
                    "INT",
                    {
                        "default": 20,
                        "min": 1,
                        "max": 1000,
                        "tooltip": "Total sampling steps (for step-based scheduling like [thing:10])",
                    },
                ),
                "normalization": (
                    "BOOLEAN",
                    {"default": False, "label_on": "Enable", "label_off": "Disable"},
                ),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "conditioning/advanced"

    def encode(self, clip, text, steps=20, normalization=False):
        """
        Main encode function - A1111 style with step-based scheduling.

        Key features:
        - BREAK segments are tokenized SEPARATELY for isolation
        - Each BREAK segment becomes its own 77-token batch
        - Scheduling/alternation evaluated per-step like A1111
        """
        if clip is None:
            raise RuntimeError("ERROR: clip input is None")

        is_sdxl = hasattr(clip.cond_stage_model, "clip_l") and hasattr(
            clip.cond_stage_model, "clip_g"
        )

        logger.info(f"[God Node] ========== DEBUG START ==========")
        logger.info(
            f"[God Node] Input text: {text[:100]}..."
            if len(text) > 100
            else f"[God Node] Input text: {text}"
        )
        logger.info(f"[God Node] Model: {'SDXL' if is_sdxl else 'SD1.5'}")
        logger.info(f"[God Node] Steps: {steps}")
        logger.info(f"[God Node] Normalization: {'ON' if normalization else 'OFF'}")

        # Use A1111-style schedule generation
        schedule = get_prompt_schedule(text, steps)

        logger.info(f"[God Node] Raw schedule ({len(schedule)} entries):")
        for end_step, prompt_text in schedule[:10]:
            preview = prompt_text[:60] + "..." if len(prompt_text) > 60 else prompt_text
            logger.info(f"  Step {end_step}: '{preview}'")

        # Group consecutive identical prompts
        grouped_schedule = self._group_schedule(schedule, steps)

        logger.info(f"[God Node] Grouped schedule ({len(grouped_schedule)} ranges):")
        for start_pct, end_pct, prompt_text in grouped_schedule:
            preview = prompt_text[:60] + "..." if len(prompt_text) > 60 else prompt_text
            logger.info(f"  {start_pct:.1%}-{end_pct:.1%}: '{preview}'")

        # Collect unique prompts
        unique_prompts = list(set(prompt for _, _, prompt in grouped_schedule))
        logger.info(f"[God Node] Unique prompts to encode: {len(unique_prompts)}")

        logger.info(f"[God Node] ========== DEBUG END ==========")

        # Encode all unique prompts WITH BREAK ISOLATION
        encoded_cache = {}
        for prompt_text in unique_prompts:
            if prompt_text in encoded_cache:
                continue

            # Handle BREAK segments with proper isolation
            cond, pooled = self._encode_with_break_isolation(
                clip, prompt_text, normalization, is_sdxl
            )
            encoded_cache[prompt_text] = (cond, pooled)

        # Build final conditioning schedule
        final_conditionings = []
        for start_pct, end_pct, prompt_text in grouped_schedule:
            if prompt_text not in encoded_cache:
                continue

            cond, pooled = encoded_cache[prompt_text]

            cond_dict = {
                "start_percent": start_pct,
                "end_percent": end_pct,
            }
            if pooled is not None:
                cond_dict["pooled_output"] = pooled

            final_conditionings.append([cond, cond_dict])

        if not final_conditionings:
            tokens = clip.tokenize("")
            result = clip.encode_from_tokens_scheduled(tokens)
            return (result,)

        return (final_conditionings,)

    def _encode_with_break_isolation(self, clip, prompt_text, normalization, is_sdxl):
        """
        Encode prompt with BREAK isolation.

        Each BREAK segment is tokenized separately, creating isolated
        77-token batches that prevent bleed-over between segments.
        """
        # Split by BREAK
        break_segments = re.split(r"\s*\bBREAK\b\s*", prompt_text)

        logger.info(f"[God Node] Encoding {len(break_segments)} BREAK segment(s)")

        if len(break_segments) == 1:
            # No BREAK, encode normally
            tokens = clip.tokenize(break_segments[0])
            return self._encode_with_direct_scaling(
                clip, tokens, normalization, is_sdxl
            )

        # Multiple segments - tokenize each separately for isolation
        all_tokens = None

        for i, segment in enumerate(break_segments):
            segment = segment.strip()
            if not segment:
                continue

            segment_tokens = clip.tokenize(segment)

            if all_tokens is None:
                all_tokens = segment_tokens
            else:
                # Merge token batches - each segment becomes its own batch
                for key in segment_tokens:
                    if key in all_tokens:
                        all_tokens[key] = all_tokens[key] + segment_tokens[key]
                    else:
                        all_tokens[key] = segment_tokens[key]

            logger.info(
                f"  Segment {i}: '{segment[:40]}...' -> {len(segment_tokens.get('l', segment_tokens.get('g', [])))} batch(es)"
            )

        if all_tokens is None:
            # Empty prompt
            all_tokens = clip.tokenize("")

        return self._encode_with_direct_scaling(
            clip, all_tokens, normalization, is_sdxl
        )

    def _group_schedule(self, schedule, steps):
        """
        Convert A1111-style schedule [(end_step, text), ...]
        to ComfyUI-style [(start_pct, end_pct, text), ...].
        """
        if not schedule:
            return []

        groups = []
        prev_end_step = 0

        for end_step, prompt_text in schedule:
            start_pct = prev_end_step / steps
            end_pct = end_step / steps

            if groups and groups[-1][2] == prompt_text:
                groups[-1] = (groups[-1][0], end_pct, prompt_text)
            else:
                groups.append((start_pct, end_pct, prompt_text))

            prev_end_step = end_step

        return groups

    def _encode_with_direct_scaling(self, clip, tokens, normalization, is_sdxl):
        """Encode with direct scaling (anti-burn) like A1111's EmphasisOriginalNoNorm."""
        clip.load_model()
        clip.cond_stage_model.reset_clip_options()

        if clip.layer_idx is not None:
            clip.cond_stage_model.set_clip_options({"layer": clip.layer_idx})

        device = clip.patcher.load_device
        clip.cond_stage_model.set_clip_options({"execution_device": device})

        if is_sdxl:
            weights_l = self._extract_weights(tokens.get("l", []))
            weights_g = self._extract_weights(tokens.get("g", []))
            tokens_neutral = {
                "l": self._neutralize_weights(tokens.get("l", [])),
                "g": self._neutralize_weights(tokens.get("g", [])),
            }
        else:
            weights = self._extract_weights(
                tokens.get("l", tokens) if isinstance(tokens, dict) else tokens
            )
            tokens_neutral = self._neutralize_weights(
                tokens.get("l", tokens) if isinstance(tokens, dict) else tokens
            )
            if isinstance(tokens, dict):
                tokens_neutral = {"l": tokens_neutral}

        o = clip.cond_stage_model.encode_token_weights(tokens_neutral)
        cond, pooled = o[:2]

        if is_sdxl:
            cond = self._apply_direct_scaling_sdxl(
                cond, weights_l, weights_g, normalization
            )
        else:
            cond = self._apply_direct_scaling(cond, weights, normalization)

        if pooled is not None:
            pooled = pooled.to(model_management.intermediate_device())
        cond = cond.to(model_management.intermediate_device())

        return cond, pooled

    def _extract_weights(self, token_batches):
        all_weights = []
        for batch in token_batches:
            if not batch:
                all_weights.append([])
                continue
            batch_weights = [
                t[1] if hasattr(t, "__getitem__") and len(t) > 1 else 1.0 for t in batch
            ]
            all_weights.append(batch_weights)
        return all_weights

    def _neutralize_weights(self, token_batches):
        neutralized = []
        for batch in token_batches:
            if not batch:
                neutralized.append([])
                continue
            new_batch = [
                (t[0], 1.0) if hasattr(t, "__getitem__") else (t, 1.0) for t in batch
            ]
            neutralized.append(new_batch)
        return neutralized

    def _apply_direct_scaling(self, cond, weights, normalization):
        """Apply A1111-style direct scaling to embeddings."""
        cond = cond.clone()
        for batch_idx, batch_weights in enumerate(weights):
            if batch_idx >= cond.shape[0]:
                break
            for token_idx, weight in enumerate(batch_weights):
                if token_idx >= cond.shape[1]:
                    break
                if weight != 1.0:
                    if normalization:
                        # EmphasisOriginal - normalize mean after scaling
                        mean_before = cond[batch_idx, token_idx].mean()
                        cond[batch_idx, token_idx] *= weight
                        mean_after = cond[batch_idx, token_idx].mean()
                        if mean_after != 0:
                            cond[batch_idx, token_idx] *= mean_before / mean_after
                    else:
                        # EmphasisOriginalNoNorm - just scale
                        cond[batch_idx, token_idx] *= weight
        return cond

    def _apply_direct_scaling_sdxl(self, cond, weights_l, weights_g, normalization):
        """SDXL output is [batch, seq, 2048] where 2048 = 768 (L) + 1280 (G)."""
        cond = cond.clone()
        dim_l = 768
        cond_l = cond[:, :, :dim_l]
        cond_g = cond[:, :, dim_l:]

        cond_l = self._apply_direct_scaling(cond_l, weights_l, normalization)
        cond_g = self._apply_direct_scaling(cond_g, weights_g, normalization)

        return torch.cat([cond_l, cond_g], dim=-1)


NODE_CLASS_MAPPINGS = {"A1111Prompt": A1111PromptNode}
NODE_DISPLAY_NAME_MAPPINGS = {"A1111Prompt": "A1111 Style Prompt (God Node)"}
