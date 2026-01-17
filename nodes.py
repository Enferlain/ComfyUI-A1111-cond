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
from .hooks import create_step_schedule_cond

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
                "model": ("MODEL",),
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
                "debug": (
                    "BOOLEAN",
                    {"default": False, "label_on": "Enable", "label_off": "Disable"},
                ),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "MODEL")
    RETURN_NAMES = ("conditioning", "model")
    FUNCTION = "encode"
    CATEGORY = "conditioning/advanced"

    def encode(
        self, clip, text, model=None, steps=20, normalization=False, debug=False
    ):
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

        if debug:
            logger.info(f"[A1111 Prompt] ========== DEBUG START ==========")
            logger.info(
                f"[A1111 Prompt] Input text: {text[:100]}..."
                if len(text) > 100
                else f"[A1111 Prompt] Input text: {text}"
            )
            logger.info(f"[A1111 Prompt] Model: {'SDXL' if is_sdxl else 'SD1.5'}")
            logger.info(f"[A1111 Prompt] Steps: {steps}")
            logger.info(
                f"[A1111 Prompt] Normalization: {'ON' if normalization else 'OFF'}"
            )

        schedule = get_prompt_schedule(text, steps)

        # Check if we need alternation hook (if any step has alternation)
        # The parser returns a flat list of (end_step, prompt)
        # If the prompts are DIFFERENT for almost every step, it's likely alternation or a gradient
        # We can just check if we have many changes.

        # Actually, let's just checking if the schedule is complex.
        # But wait, [A|B] generates a schedule that changes every step.
        # So we can detect if len(schedule) is roughly equal to steps?
        # Or simpler: always use the hook if there's more than 1 item in schedule?
        # NO. Standard scheduling [from:to:when] creates few segments.
        # Alternation [A|B] creates 'steps' segments.

        # If we just implemented the hook for EVERYTHING, would that be bad?
        # It adds a python call per step. Might be slightly slower.
        # But it guarantees correctness.
        # However, standard ComfyUI scheduling (TIMESTEP range) is more efficient for simple [from:to].
        # Let's support both.

        # Strategy:
        # 1. Flatten schedule to per-step list of prompts (size = steps)
        # 2. Encode all unique prompts
        # 3. Create a list of (cond, pooled) for each step
        # 4. Attach hook with this list

        # First, expand schedule to full per-step list
        full_step_prompts = [""] * steps
        prev_end = 0
        for end_step, prompt in schedule:
            # fill from prev_end to end_step
            # clamp just in case
            safe_end = min(end_step, steps)
            for i in range(prev_end, safe_end):
                full_step_prompts[i] = prompt
            prev_end = safe_end

        # Analyze if we really need per-step switching
        # Count transitions
        transitions = 0
        for i in range(1, len(full_step_prompts)):
            if full_step_prompts[i] != full_step_prompts[i - 1]:
                transitions += 1

        # If few transitions (like < 5?), we can use standard Comfy scheduling ranges for efficiency.
        # If many transitions (alternation [A|B] = steps/2 transitions), use Hook.
        # Let's say if transitions > 4, or if user asks for it (we can add a force/debug toggle later).
        # For now, let's be aggressive and use Hook for > 2 transitions to be safe for [A|B|C].

        # TODO: A1111 "Scheduling" [from:to:when] is also technically 'step based'.
        # ComfyUI Percentages are sigma-based approximations.
        # To get TRUE A1111 parity for even simple scheduling, we should use the Hook!
        # Because ComfyUI's percent->sigma mapping is non-linear and might drift from A1111's linear step count.
        # So, ALWAYS USE HOOK if there is ANY schedule?

        has_schedule = len(schedule) > 1

        if debug:
            unique_prompts = list(set(full_step_prompts))
            logger.info(
                f"[A1111 Prompt] Unique prompts: {len(unique_prompts)} (will encode each once)"
            )
            logger.info(
                f"[A1111 Prompt] Step transitions: {transitions} across {steps} steps"
            )

            # Show alternation pattern if applicable
            if transitions > steps // 3:  # Likely alternation
                # Sample a few steps to show pattern
                sample_steps = [0, 1, 2, steps // 2, steps - 1]
                pattern = []
                for s in sample_steps:
                    if s < len(full_step_prompts):
                        # Extract just the varying part (first 50 chars)
                        p = full_step_prompts[s][:50]
                        pattern.append(f"Step {s}: {p}...")
                logger.info("[A1111 Prompt] Alternation pattern sample:")
                for p in pattern:
                    logger.info(f"  {p}")
            else:
                # Show range-based transitions
                logger.info("[A1111 Prompt] Schedule segments:")
                for seg in schedule[:5]:  # First 5 segments
                    end_step, prompt = seg
                    logger.info(f"  Until step {end_step}: {prompt[:60]}...")

        if not has_schedule:
            # Just one constant prompt - no step switching needed
            prompt_text = schedule[0][1] if schedule else ""
            cond, pooled = self._encode_with_break_isolation(
                clip, prompt_text, normalization, is_sdxl, debug
            )
            return ([[cond, {"pooled_output": pooled}]], model)

        # Use step-based conditioning for strict A1111 parity
        encoded_cache = {}
        unique_prompts = list(set(full_step_prompts))

        for prompt_text in unique_prompts:
            if prompt_text in encoded_cache:
                continue
            cond, pooled = self._encode_with_break_isolation(
                clip, prompt_text, normalization, is_sdxl, debug
            )
            encoded_cache[prompt_text] = (cond, pooled)

        # Find maximum sequence length across all embeddings
        max_seq_len = max(cond.shape[1] for cond, _ in encoded_cache.values())

        # Pad all embeddings to the same length
        for prompt_text in encoded_cache:
            cond, pooled = encoded_cache[prompt_text]
            if cond.shape[1] < max_seq_len:
                pad_size = max_seq_len - cond.shape[1]
                padding = torch.zeros(
                    cond.shape[0],
                    pad_size,
                    cond.shape[2],
                    device=cond.device,
                    dtype=cond.dtype,
                )
                cond = torch.cat([cond, padding], dim=1)
                encoded_cache[prompt_text] = (cond, pooled)
                if debug:
                    logger.info(
                        f"[A1111 Prompt] Padded embedding from {cond.shape[1] - pad_size} to {max_seq_len}"
                    )

        # Build per-step embedding list
        step_embeddings = []
        for prompt in full_step_prompts:
            step_embeddings.append(encoded_cache[prompt])

        # Use the first step's conditioning as base
        base_cond, base_pooled = step_embeddings[0]

        # Create conditioning with step schedule attached
        conditioning = create_step_schedule_cond(
            step_embeddings, steps, base_cond, base_pooled
        )

        # If model is provided, set up the wrapper for step-based conditioning
        if model is not None:
            from .hooks import setup_step_conditioning_on_model

            model = model.clone()  # Don't modify the original
            setup_step_conditioning_on_model(model, step_embeddings, steps)
            if debug:
                logger.info(
                    f"[A1111 Prompt] Set up step conditioning on model for {len(step_embeddings)} steps"
                )

        return (conditioning, model)

    def _encode_with_break_isolation(
        self, clip, prompt_text, normalization, is_sdxl, debug
    ):
        """
        Encode prompt with BREAK isolation.

        Each BREAK segment is tokenized separately, creating isolated
        77-token batches that prevent bleed-over between segments.
        """
        # Split by BREAK
        break_segments = re.split(r"\s*\bBREAK\b\s*", prompt_text)

        if debug:
            logger.info(
                f"[A1111 Prompt] Encoding {len(break_segments)} BREAK segment(s)"
            )

        if len(break_segments) == 1:
            # No BREAK, encode normally
            tokens = clip.tokenize(break_segments[0])
            return self._encode_with_direct_scaling(
                clip, tokens, normalization, is_sdxl, debug
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

            if debug:
                logger.info(
                    f"  Segment {i}: '{segment[:40]}...' -> {len(segment_tokens.get('l', segment_tokens.get('g', [])))} batch(es)"
                )

        if all_tokens is None:
            # Empty prompt
            all_tokens = clip.tokenize("")

        return self._encode_with_direct_scaling(
            clip, all_tokens, normalization, is_sdxl, debug
        )

    def _encode_with_direct_scaling(
        self, clip, tokens, normalization, is_sdxl, debug=False
    ):
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
            tokens_neutral = tokens.copy()
            tokens_neutral["l"] = self._neutralize_weights(tokens.get("l", []))
            tokens_neutral["g"] = self._neutralize_weights(tokens.get("g", []))
        else:
            weights = self._extract_weights(
                tokens.get("l", tokens) if isinstance(tokens, dict) else tokens
            )
            tokens_neutral = self._neutralize_weights(
                tokens.get("l", tokens) if isinstance(tokens, dict) else tokens
            )
            if isinstance(tokens, dict):
                tokens_neutral_dict = tokens.copy()
                tokens_neutral_dict["l"] = tokens_neutral
                tokens_neutral = tokens_neutral_dict

        o = clip.cond_stage_model.encode_token_weights(tokens_neutral)
        cond, pooled = o[:2]

        if is_sdxl:
            cond = self._apply_direct_scaling_sdxl(
                cond, weights_l, weights_g, normalization, debug
            )
        else:
            cond = self._apply_direct_scaling(cond, weights, normalization, debug)

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

    def _apply_direct_scaling(self, cond, weights, normalization, debug=False):
        """
        Apply A1111-style direct scaling to embeddings.

        Uses proper tensor broadcasting like A1111's emphasis.py:
        self.z = self.z * self.multipliers.reshape(self.multipliers.shape + (1,)).expand(self.z.shape)

        Note: weights is List[List[float]] where each inner list is a BREAK chunk.
        cond shape is [batch, seq, dim] where seq = sum of all chunk lengths.
        We need to flatten weights to match the sequence dimension.
        """
        cond = cond.clone()

        # Flatten weights from all chunks into a single sequence
        # weights is List[List[float]] - one list per BREAK chunk (or token batch)
        flat_weights = []
        for chunk_weights in weights:
            flat_weights.extend(chunk_weights)

        # Match to cond's sequence length
        seq_len = cond.shape[1]
        if len(flat_weights) < seq_len:
            flat_weights = flat_weights + [1.0] * (seq_len - len(flat_weights))
        elif len(flat_weights) > seq_len:
            flat_weights = flat_weights[:seq_len]

        # Create tensor matching cond's batch dimension [batch, seq]
        batch_size = cond.shape[0]
        multipliers = torch.tensor(
            [flat_weights] * batch_size, device=cond.device, dtype=cond.dtype
        )

        # Debug: show weight statistics
        if debug:
            non_unity = [w for w in flat_weights if abs(w - 1.0) > 0.001]
            if non_unity:
                logger.info(
                    f"[A1111 Prompt] Emphasis weights applied: {len(non_unity)} tokens modified"
                )
                logger.info(
                    f"[A1111 Prompt]   Weight range: [{min(non_unity):.3f}, {max(non_unity):.3f}]"
                )
                # Show first few modified weights with positions
                weight_samples = [
                    (i, w) for i, w in enumerate(flat_weights) if abs(w - 1.0) > 0.001
                ][:5]
                logger.info(f"[A1111 Prompt]   Sample weights: {weight_samples}")
            else:
                logger.info(f"[A1111 Prompt] No emphasis weights (all tokens = 1.0)")

        if normalization:
            # A1111 EmphasisOriginal: preserve original mean
            original_mean = cond.mean()

        # A1111-style broadcasting: expand [batch, seq] -> [batch, seq, dim]
        multipliers = multipliers.unsqueeze(-1).expand_as(cond)
        cond = cond * multipliers

        if normalization:
            new_mean = cond.mean()
            if new_mean.abs() > 1e-8:
                scale_factor = original_mean / new_mean
                cond = cond * scale_factor
                if debug:
                    logger.info(
                        f"[A1111 Prompt] Normalization applied: scale={scale_factor.item():.4f}"
                    )

        return cond

    def _apply_direct_scaling_sdxl(
        self, cond, weights_l, weights_g, normalization, debug=False
    ):
        """SDXL output is [batch, seq, 2048] where 2048 = 768 (L) + 1280 (G)."""
        if cond.shape[-1] != 2048:
            # Fallback for non-SDXL models (e.g. SD3) that might trigger is_sdxl check
            logger.warning(
                f"[A1111 Prompt] Unexpected embedding dimension {cond.shape[-1]} (expected 2048 for SDXL). Applying symmetric scaling."
            )
            return self._apply_direct_scaling(
                cond, weights_g if weights_g else weights_l, normalization, debug
            )

        if debug:
            logger.info(f"[A1111 Prompt] SDXL scaling: cond shape {cond.shape}")
            logger.info(
                f"[A1111 Prompt]   Applying weights to CLIP-L (768 dims) AND CLIP-G (1280 dims)"
            )

        cond = cond.clone()
        dim_l = 768
        cond_l = cond[:, :, :dim_l]
        cond_g = cond[:, :, dim_l:]

        # A1111 scales BOTH CLIP-L and CLIP-G embeddings
        if debug:
            logger.info(f"[A1111 Prompt]   Scaling CLIP-L...")
        cond_l = self._apply_direct_scaling(cond_l, weights_l, normalization, debug)
        if debug:
            logger.info(f"[A1111 Prompt]   Scaling CLIP-G...")
        cond_g = self._apply_direct_scaling(cond_g, weights_g, normalization, debug)

        return torch.cat([cond_l, cond_g], dim=-1)


NODE_CLASS_MAPPINGS = {
    "A1111Prompt": A1111PromptNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "A1111Prompt": "A1111 Style Prompt",
}
