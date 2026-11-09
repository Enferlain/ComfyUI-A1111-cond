"""Experimental attention-odds weighting after Anima's key normalization."""

from contextlib import contextmanager


def biased_attention(original, weights):
    import torch
    import torch.nn.functional as F

    weights = torch.as_tensor(weights, dtype=torch.float32)
    if weights.ndim != 1 or not bool(torch.isfinite(weights).all()) or not bool((weights > 0).all()):
        raise ValueError("Attention-bias weights must be a finite positive vector.")

    def apply(q, k, v, transformer_options=None):
        options = transformer_options or {}
        groups = options.get("cond_or_uncond")
        if bool((weights == 1).all()):
            return original(q, k, v, transformer_options=options)
        if not groups or any(group not in (0, 1) for group in groups) or q.shape[0] % len(groups):
            raise ValueError("Cannot identify positive/negative attention batch groups.")
        if 0 not in groups:
            return original(q, k, v, transformer_options=options)
        if k.shape[1] != len(weights):
            raise ValueError("Attention weights do not match the prepared context length.")
        batch_per_group = q.shape[0] // len(groups)
        bias = torch.zeros((q.shape[0], 1, 1, k.shape[1]), device=q.device, dtype=q.dtype)
        log_weights = weights.to(device=q.device, dtype=q.dtype).log()
        for index, group in enumerate(groups):
            if group == 0:
                bias[index * batch_per_group:(index + 1) * batch_per_group] = log_weights
        # Mixed batches use this research SDPA backend for both branches.
        # Negative rows have zero bias, but bitwise parity with a different
        # optimized original backend is not asserted.
        output = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=bias
        )
        return output.transpose(1, 2).flatten(2)

    return apply


@contextmanager
def temporary_attention_bias(model, weights):
    originals = []
    try:
        for block in model.model.diffusion_model.blocks:
            attention = block.cross_attn
            originals.append((attention, attention.attn_op))
            attention.attn_op = biased_attention(attention.attn_op, weights)
        yield
    finally:
        for attention, original in originals:
            attention.attn_op = original
