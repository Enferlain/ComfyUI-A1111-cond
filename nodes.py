"""
A1111 Style Prompt Node (God Node)

Implements A1111-style prompt handling:
- Hard chunking with padding (isolation/no bleed)
- Direct scaling (no burn) instead of interpolation
- BREAK support
- Scheduling [from:to:when] (Switch, Add, Remove)
- Alternation [A|B] (via Blending)

TODO (Future Work):
- [ ] Token counter widget (requires frontend access)
- [ ] True per-step alternation via sampler hooks
- [ ] Textual Inversion embedding support validation
- [ ] SD3 triple-encoder support
"""

import torch
import logging
import comfy.model_management as model_management
from .parser import parse_a1111_prompt

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
        Main encode function.

        Args:
            clip: CLIP model
            text: Prompt text with A1111 syntax
            steps: Total sampling steps (for step-based scheduling)
            normalization: Enable EmphasisOriginalNoNorm behavior
        """
        if clip is None:
            raise RuntimeError("ERROR: clip input is None")

        chunks = parse_a1111_prompt(text, steps=steps)

        is_sdxl = hasattr(clip.cond_stage_model, "clip_l") and hasattr(
            clip.cond_stage_model, "clip_g"
        )

        logger.info(f"[God Node] BREAK segments: {len(chunks)}")
        logger.info(f"[God Node] Model: {'SDXL' if is_sdxl else 'SD1.5'}")
        logger.info(f"[God Node] Steps: {steps}")
        logger.info(f"[God Node] Normalization: {'ON' if normalization else 'OFF'}")

        time_ranges = set()
        for break_chunk in chunks:
            for part in break_chunk:
                time_ranges.add((part.start_percent, part.end_percent))
        time_ranges = sorted(list(time_ranges))

        if len(time_ranges) > 1:
            logger.info(f"[God Node] Scheduling: {len(time_ranges)} time ranges")

        final_conditionings = []

        for start_pct, end_pct in time_ranges:
            chunk_variants = []

            for break_chunk in chunks:
                # Active parts for this slice
                slice_parts = [
                    p
                    for p in break_chunk
                    if p.start_percent <= start_pct + 0.0001
                    and p.end_percent >= end_pct - 0.0001
                ]

                if not slice_parts:
                    chunk_variants.append([("", 1.0)])
                    continue

                current_branches = [("", 1.0)]

                for p in slice_parts:
                    next_branches = []

                    if p.is_alternating:
                        options = p.alternating_options
                        n_opts = len(options)
                        base_weight = p.weight

                        for branch_text, branch_scale in current_branches:
                            for opt in options:
                                w_text = (
                                    f"({opt}:{base_weight})"
                                    if base_weight != 1.0
                                    else opt
                                )
                                new_text = (branch_text + " " + w_text).strip()
                                new_scale = branch_scale * (1.0 / n_opts)
                                next_branches.append((new_text, new_scale))
                    else:
                        w_text = f"({p.text}:{p.weight})" if p.weight != 1.0 else p.text
                        for branch_text, branch_scale in current_branches:
                            new_text = (branch_text + " " + w_text).strip()
                            next_branches.append((new_text, branch_scale))

                    current_branches = next_branches

                chunk_variants.append(current_branches)

            chunk_conds_list = []

            for variants in chunk_variants:
                # Skip empty chunks (e.g., parts not active in this time range)
                if len(variants) == 1 and variants[0][0] == "":
                    continue

                accumulated_cond = None
                accumulated_pooled = None

                for text, scale in variants:
                    tokens = clip.tokenize(text)
                    cond, pooled = self._encode_with_direct_scaling(
                        clip, tokens, normalization, is_sdxl
                    )

                    cond = cond * scale
                    if pooled is not None:
                        pooled = pooled * scale

                    if accumulated_cond is None:
                        accumulated_cond = cond
                        accumulated_pooled = pooled
                    else:
                        accumulated_cond = accumulated_cond + cond
                        if accumulated_pooled is not None:
                            accumulated_pooled = accumulated_pooled + pooled

                chunk_conds_list.append((accumulated_cond, accumulated_pooled))

            if chunk_conds_list:
                final_cond_tensor = torch.cat([x[0] for x in chunk_conds_list], dim=1)
                final_pooled = chunk_conds_list[0][1]

                cond_dict = {
                    "start_percent": start_pct,
                    "end_percent": end_pct,
                }
                if final_pooled is not None:
                    cond_dict["pooled_output"] = final_pooled

                final_conditionings.append([final_cond_tensor, cond_dict])

        if not final_conditionings:
            tokens = clip.tokenize("")
            result = clip.encode_from_tokens_scheduled(tokens)
            return (result,)

        return (final_conditionings,)

    def _encode_with_direct_scaling(self, clip, tokens, normalization, is_sdxl):
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
            # SD1.5: tokens is a dict {"l": [...]}
            # FIXME: Verify SD1.5 token format - may need tokens.get("l", tokens)
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
        cond = cond.clone()
        for batch_idx, batch_weights in enumerate(weights):
            if batch_idx >= cond.shape[0]:
                break
            for token_idx, weight in enumerate(batch_weights):
                if token_idx >= cond.shape[1]:
                    break
                if weight != 1.0:
                    if normalization:
                        mean_before = cond[batch_idx, token_idx].mean()
                        cond[batch_idx, token_idx] *= weight
                        mean_after = cond[batch_idx, token_idx].mean()
                        if mean_after != 0:
                            cond[batch_idx, token_idx] *= mean_before / mean_after
                    else:
                        cond[batch_idx, token_idx] *= weight
        return cond

    def _apply_direct_scaling_sdxl(self, cond, weights_l, weights_g, normalization):
        """
        SDXL output is [batch, seq, 2048] where 2048 = 768 (L) + 1280 (G).
        We apply weights to each encoder's output independently.

        NOTE: L and G tokenizations may differ slightly in edge cases.
        Currently we assume they produce same token counts per word.
        """
        cond = cond.clone()
        dim_l = 768
        cond_l = cond[:, :, :dim_l]
        cond_g = cond[:, :, dim_l:]

        cond_l = self._apply_direct_scaling(cond_l, weights_l, normalization)
        cond_g = self._apply_direct_scaling(cond_g, weights_g, normalization)

        return torch.cat([cond_l, cond_g], dim=-1)


NODE_CLASS_MAPPINGS = {"A1111Prompt": A1111PromptNode}
NODE_DISPLAY_NAME_MAPPINGS = {"A1111Prompt": "A1111 Style Prompt (God Node)"}
