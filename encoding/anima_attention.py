"""Conditioning-scoped Anima attention bias, without changing model objects."""

import torch
from comfy.hooks import HookGroup, TransformerOptionsHook, EnumHookScope


class AnimaAttention:
    """Install invocation-local cross-attention markers and a log-weight mask.

    Cosmos calls attn2 patches before projection and the attention override after
    key normalization. Marking the call avoids guessing from sequence lengths.
    HookedOnly makes Comfy batch this conditioning separately from other prompts.
    """

    def __init__(self, weights):
        self.weights = torch.as_tensor(weights, dtype=torch.float32).clone()
        if self.weights.ndim != 1 or not bool(torch.isfinite(self.weights).all()) or not bool((self.weights > 0).all()):
            raise ValueError("Anima attention mode requires finite, strictly positive weights. Use single_pass or native for zero/negative weights.")

    def __call__(self, executor, *args, **kwargs):
        from comfy.ldm.anima.model import Anima

        if not isinstance(executor.class_obj, Anima):
            raise ValueError("Anima attention conditioning requires an Anima diffusion model.")
        options = dict(kwargs.get("transformer_options", {}))
        if options.get("a1111_anima_attention"):
            raise ValueError("Combining multiple Anima attention hooks on one conditioning is unsupported.")
        options["a1111_anima_attention"] = True
        state = {"cross": False, "calls": 0, "patches": 0, "last_cross": False}
        biases = {}

        def mark_self(*args, **kwargs):
            state["cross"] = False
            state["last_cross"] = False
            return {}

        def mark_cross(*args, **kwargs):
            state["cross"] = True
            state["last_cross"] = True
            state["patches"] += 1
            return {}

        patches = dict(options.get("patches", {}))
        patches["attn1_patch"] = list(patches.get("attn1_patch", ())) + [mark_self]
        patches["attn2_patch"] = list(patches.get("attn2_patch", ())) + [mark_cross]
        options["patches"] = patches
        previous = options.get("optimized_attention_override")

        def attention(original, q, k, v, heads, **attn_kwargs):
            cross = state["cross"]
            if state["last_cross"] and not cross:
                raise ValueError("Unexpected repeated Anima cross-attention call. Update the node or select native.")
            state["cross"] = False
            if cross:
                state["calls"] += 1
                if not attn_kwargs.get("skip_reshape") or k.ndim != 4:
                    raise ValueError("Unexpected Anima attention layout; update ComfyUI or use native mode.")
                # Anima.preprocess_text_embeds pads T5 adapter output to 512.
                expected = max(512, len(self.weights))
                if k.shape[-2] != expected:
                    raise ValueError("Anima attention weights do not match the prepared context length.")
                key = (q.device, q.dtype)
                if key not in biases:
                    weights = self.weights.to(device=q.device)
                    biases[key] = torch.nn.functional.pad(weights.log(), (0, expected - len(weights))).to(q.dtype)[None, None, None, :]
                bias = biases[key]
                mask = attn_kwargs.get("mask")
                if mask is not None:
                    mask = mask.to(device=q.device)
                    if mask.ndim == 2:
                        mask = mask.unsqueeze(0)
                    if mask.ndim == 3:
                        mask = mask.unsqueeze(1)
                    if mask.dtype == torch.bool:
                        mask = torch.zeros_like(mask, dtype=q.dtype).masked_fill(~mask, float("-inf"))
                    else:
                        mask = mask.to(dtype=q.dtype)
                    bias = bias + mask
                attn_kwargs["mask"] = bias
            # Keep the installed attention backend and any prior override. Mask
            # support/fallback is handled by ComfyUI, including output layout.
            target = previous if previous is not None else lambda f, *a, **kw: f(*a, **kw)
            return target(original, q, k, v, heads, **attn_kwargs)

        options["optimized_attention_override"] = attention
        kwargs["transformer_options"] = options
        result = executor(*args, **kwargs)
        if not state["calls"]:
            raise ValueError("This ComfyUI attention implementation does not support Anima attention mode. Update ComfyUI or select native.")
        if state["calls"] != state["patches"]:
            raise ValueError("Anima attention patch/call mismatch. Update the node or select native.")
        return result


def attention_hooks(weights):
    attention = AnimaAttention(weights)
    if bool((attention.weights == 1).all()):
        return None
    hooks = HookGroup()
    hooks.add(TransformerOptionsHook(
        transformers_dict={"wrappers": {"diffusion_model": {"a1111_anima_attention": [attention]}}},
        hook_scope=EnumHookScope.HookedOnly,
    ))
    return hooks
