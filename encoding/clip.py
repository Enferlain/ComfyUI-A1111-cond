"""CLIP-family chunk isolation and A1111 direct embedding weights."""

import logging
import re

import torch
import comfy.model_management as model_management

from .base import EncoderCapabilities, PromptEncoder

logger = logging.getLogger("A1111PromptNode")


class CLIPEncoder(PromptEncoder):
    capabilities = EncoderCapabilities(breaks=True, scheduling=True, normalization=True)

    def __init__(self, channels):
        self.channels = channels
        self.name = "SDXL CLIP-L/G" if channels == ("l", "g") else "SD1.5 CLIP"

    def matches(self, clip, streams):
        return streams == set(self.channels)

    def _tokenize(self, clip, text):
        tokens = clip.tokenize(text)
        if not isinstance(tokens, dict) or not self.matches(clip, set(tokens)):
            raise ValueError(f"{self.name} returned unexpected token streams.")
        return tokens

    def encode(self, clip, text, normalization=False, debug=False):
        self.validate_request(text, normalization=normalization)
        segments = re.split(r"\s*\bBREAK\b\s*", text)
        if len(segments) == 1:
            tokens = self._tokenize(clip, text)
        else:
            tokens = None
            for segment in segments:
                if not segment.strip():
                    continue
                segment_tokens = self._tokenize(clip, segment.strip())
                if tokens is None:
                    tokens = segment_tokens
                else:
                    tokens = {key: tokens[key] + segment_tokens[key] for key in self.channels}
            if tokens is None:
                tokens = self._tokenize(clip, "")
        return self._encode_with_direct_scaling(clip, tokens, normalization, debug)

    def _encode_with_direct_scaling(
        self, clip, tokens, normalization, debug=False
    ):
        """Encode with direct scaling (anti-burn) like A1111's EmphasisOriginalNoNorm."""
        clip.load_model()
        clip.cond_stage_model.reset_clip_options()

        if clip.layer_idx is not None:
            clip.cond_stage_model.set_clip_options({"layer": clip.layer_idx})

        device = clip.patcher.load_device
        clip.cond_stage_model.set_clip_options({"execution_device": device})

        if self.channels == ("l", "g"):
            weights_l = self._extract_weights(tokens.get("l", []))
            weights_g = self._extract_weights(tokens.get("g", []))
            tokens_neutral = tokens.copy()
            tokens_neutral["l"] = self._neutralize_weights(tokens.get("l", []))
            tokens_neutral["g"] = self._neutralize_weights(tokens.get("g", []))
        else:
            weights = self._extract_weights(tokens["l"])
            tokens_neutral = {"l": self._neutralize_weights(tokens["l"])}

        o = clip.cond_stage_model.encode_token_weights(tokens_neutral)
        cond, pooled = o[:2]

        if self.channels == ("l", "g"):
            cond = self._apply_direct_scaling_sdxl(
                cond, weights_l, weights_g, normalization, debug
            )
        else:
            cond = self._apply_direct_scaling(cond, weights, normalization, debug)

        if pooled is not None:
            pooled = pooled.to(model_management.intermediate_device())
        cond = cond.to(model_management.intermediate_device())

        metadata = dict(o[2]) if len(o) > 2 else {}
        metadata["pooled_output"] = pooled
        return cond, metadata

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
                logger.info("[A1111 Prompt] No emphasis weights (all tokens = 1.0)")

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
            raise ValueError(
                f"SDXL CLIP-L/G requires 2048 embedding dimensions; got {cond.shape[-1]}. "
                "This conditioning layout needs a separate encoder implementation."
            )

        if debug:
            logger.info(f"[A1111 Prompt] SDXL scaling: cond shape {cond.shape}")
            logger.info(
                "[A1111 Prompt]   Applying weights to CLIP-L (768 dims) AND CLIP-G (1280 dims)"
            )

        cond = cond.clone()
        dim_l = 768
        cond_l = cond[:, :, :dim_l]
        cond_g = cond[:, :, dim_l:]

        # A1111 scales BOTH CLIP-L and CLIP-G embeddings
        if debug:
            logger.info("[A1111 Prompt]   Scaling CLIP-L...")
        cond_l = self._apply_direct_scaling(cond_l, weights_l, normalization, debug)
        if debug:
            logger.info("[A1111 Prompt]   Scaling CLIP-G...")
        cond_g = self._apply_direct_scaling(cond_g, weights_g, normalization, debug)

        return torch.cat([cond_l, cond_g], dim=-1)
