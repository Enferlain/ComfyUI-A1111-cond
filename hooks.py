"""
A1111 Step-Based Conditioning

Implements true A1111-style step-based prompt scheduling and alternation.

This module provides:
1. A1111StepConditioningHook - TransformerOptionsHook that swaps conditioning per-step
2. Helper functions for creating step-scheduled conditioning

The hook is attached to conditioning (not model) and automatically receives
sample_sigmas during sampling, eliminating the need for MODEL input.
"""

import math
import torch
from comfy.hooks import (
    TransformerOptionsHook,
    HookGroup,
    EnumHookScope,
    set_hooks_for_conditioning,
)
import logging

logger = logging.getLogger("A1111PromptNode")


class A1111StepConditioningHook(TransformerOptionsHook):
    """
    Hook that swaps conditioning per-step without requiring MODEL input.

    Attached to conditioning output, this hook receives sample_sigmas
    during sampling and uses it to determine current step and swap embeddings.
    """

    def __init__(
        self, step_embeddings, default_steps=28, debug=False, shared_cache=None
    ):
        """
        Args:
            step_embeddings: List of (cond_tensor, pooled_tensor) per step index
            default_steps: Original steps used to parse symbols (for scaling)
            debug: Whether to log verbose scheduling information
            shared_cache: Optional dict to share between hook clones
        """
        super().__init__(hook_scope=EnumHookScope.AllConditioning)
        self.step_embeddings = step_embeddings
        self.default_steps = default_steps
        self.debug = debug
        self._last_logged_step = -1
        self._first_swap_logged = False  # Track if we've logged the first swap

        # Share cache across clones to prevent performance regression in 2nd order samplers
        self._swap_cache = shared_cache if shared_cache is not None else {}
        self.transformers_dict = {}  # Required by base class on_apply_hooks

    def add_hook_patches(self, model, model_options, target_dict, registered):
        """Override to set model_function_wrapper directly on model_options."""
        if not self.should_register(model, model_options, target_dict, registered):
            return False

        # Check if there's already a wrapper - we need to chain them
        existing_wrapper = model_options.get("model_function_wrapper")
        if existing_wrapper is not None:
            if self.debug:
                logger.warning(
                    "[A1111 Hook] Found existing model_function_wrapper, will chain them"
                )

            # Create a chained wrapper
            def chained_wrapper(apply_model_func, args):
                # Call our wrapper first, which will call the existing one
                return self.model_function_wrapper(
                    apply_model_func, args, existing_wrapper
                )

            model_options["model_function_wrapper"] = chained_wrapper
        else:
            if self.debug:
                logger.info(
                    "[A1111 Hook] Registering model_function_wrapper on model_options"
                )
            model_options["model_function_wrapper"] = self.model_function_wrapper

        registered.add(self)
        return True

    def get_step_from_sigma(self, sigma_val, sample_sigmas):
        """
        Determine which step index we're at based on current sigma.

        Sigmas decrease during sampling, so we find which range sigma_val falls into.
        """
        if sample_sigmas is None or len(sample_sigmas) == 0:
            return 0

        num_sigmas = len(sample_sigmas)

        for i in range(num_sigmas - 1):
            s_start = (
                sample_sigmas[i].item()
                if isinstance(sample_sigmas[i], torch.Tensor)
                else sample_sigmas[i]
            )
            s_end = (
                sample_sigmas[i + 1].item()
                if isinstance(sample_sigmas[i + 1], torch.Tensor)
                else sample_sigmas[i + 1]
            )

            if s_start >= sigma_val > s_end:
                return i

            if abs(s_start - sigma_val) < 1e-4:
                return i

        last_sig = (
            sample_sigmas[-1].item()
            if isinstance(sample_sigmas[-1], torch.Tensor)
            else sample_sigmas[-1]
        )
        if sigma_val <= last_sig + 1e-4:
            return num_sigmas - 2

        return 0

    def model_function_wrapper(self, apply_model_func, args, existing_wrapper=None):
        """
        Wrapper function that intercepts model application and swaps conditioning.

        This is called by ComfyUI's sampling code when the hook is active.

        Args:
            apply_model_func: The original model function
            args: Arguments dict with input, timestep, c, cond_or_uncond
            existing_wrapper: Optional existing wrapper to chain
        """
        input_x = args["input"]
        timestep = args["timestep"]
        c = args["c"]
        cond_or_uncond = args["cond_or_uncond"]

        # 1. Quick Check: Is there even a positive conditioning in this batch to swap?
        # Standard runs have cond_or_uncond = [0, 1]. If it's just [1], skip work.
        if 0 not in cond_or_uncond or "c_crossattn" not in c:
            if existing_wrapper is not None:
                return existing_wrapper(apply_model_func, args)
            return apply_model_func(input_x, timestep, **c)

        # 2. Step detection
        sigma_val = timestep.item() if isinstance(timestep, torch.Tensor) else timestep

        # Get sample_sigmas from transformer_options in c
        transformer_options = c.get("transformer_options", {})
        sample_sigmas = transformer_options.get("sample_sigmas")

        if sample_sigmas is None:
            # No sigmas available - shouldn't happen, but fallback to first step
            if self.debug:
                logger.warning("[A1111 Hook] No sample_sigmas found, using first step")
            if existing_wrapper is not None:
                return existing_wrapper(apply_model_func, args)
            return apply_model_func(input_x, timestep, **c)

        # Calculate actual total steps from sigmas
        actual_steps = len(sample_sigmas) - 1

        # Determine current step from sigma (this returns the index in the CURRENT sampler's sigmas)
        raw_step_idx = self.get_step_from_sigma(sigma_val, sample_sigmas)

        # Scale the actual sampler step index into the range of our prepared embeddings.
        # This ensures that schedules (like transitions at 50% path) are accurate even if
        # the sampler is running with more/fewer steps than our default (usually 28).
        emb_steps = len(self.step_embeddings) - 1
        if actual_steps > 0:
            # Linear scaling: (current_sampler_step / total_sampler_steps) * total_embedding_steps
            step_idx = round(raw_step_idx * emb_steps / actual_steps)
        else:
            step_idx = 0

        # Clamp step_idx to valid range
        step_idx = max(0, min(step_idx, emb_steps))

        # 3. Cache Check: Have we already built this specific combined tensor?
        orig_cond = c["c_crossattn"]
        # Use a stable key based on structural descriptors rather than Python id().
        # This handles cloned hooks/tensors in 2nd order samplers.
        # Key: (step_index, orig_shape, cond_mask)
        cond_mask = (
            tuple(cond_or_uncond)
            if isinstance(cond_or_uncond, list)
            else cond_or_uncond
        )
        cache_key = (step_idx, orig_cond.shape, cond_mask)

        if cache_key in self._swap_cache:
            cached_cond = self._swap_cache[cache_key]
            # Fast verification of device
            if cached_cond.device == orig_cond.device:
                # Reuse existing c if possible, or build one if we're chaining
                if existing_wrapper is None:
                    new_c = c.copy()
                    new_c["c_crossattn"] = cached_cond
                    return apply_model_func(input_x, timestep, **new_c)
                else:
                    # Chaining requires a full dict for the next wrapper
                    new_c = c.copy()
                    new_c["c_crossattn"] = cached_cond
                    return existing_wrapper(
                        apply_model_func,
                        {
                            "input": input_x,
                            "timestep": timestep,
                            "c": new_c,
                            "cond_or_uncond": cond_or_uncond,
                        },
                    )

        # 4. Processing Path: We need to build the swapped tensor
        target_cond, _target_pooled = self.step_embeddings[step_idx]

        if target_cond is None:
            if existing_wrapper is not None:
                return existing_wrapper(apply_model_func, args)
            return apply_model_func(input_x, timestep, **c)

        device = orig_cond.device
        dtype = orig_cond.dtype

        # Log on very first call to confirm wrapper is active
        if self._last_logged_step == -1 and self.debug:
            logger.info(
                "[A1111 Hook] ========== WRAPPER CALLED - HOOK IS ACTIVE =========="
            )
            if existing_wrapper is not None:
                logger.info("[A1111 Hook] Chaining with existing wrapper")
            logger.info(f"[A1111 Hook] Actual sampler steps: {actual_steps}")
            logger.info(
                f"[A1111 Hook] Embeddings prepared for: {len(self.step_embeddings)} steps"
            )

        # Fast path identity check: Skip only if the target is already the input object
        if target_cond is orig_cond:
            if existing_wrapper is not None:
                return existing_wrapper(apply_model_func, args)
            return apply_model_func(input_x, timestep, **c)

        # Build modified conditioning
        # Use target_cond.to() without clone() - to() is usually a no-op if already same
        new_cond = target_cond.to(device=device, dtype=dtype)

        target_seq_len = new_cond.shape[1]
        orig_seq_len = orig_cond.shape[1]

        if target_seq_len == orig_seq_len:
            # FAST PATH: Identical sequence lengths - just slice and replace
            modified_cond = orig_cond.clone()
            for b_idx, ct in enumerate(cond_or_uncond):
                if ct == 0:  # positive
                    modified_cond[b_idx : b_idx + 1] = new_cond[0:1]
        else:
            # COMPATIBILITY PATH: Sequence length mismatch (e.g. SDXL vs SD1.5 or custom resolutions)
            lcm_len = math.lcm(target_seq_len, orig_seq_len)

            # Expand orig_cond by repeating to LCM length
            if orig_seq_len < lcm_len:
                repeat_factor = lcm_len // orig_seq_len
                expanded_orig = orig_cond.repeat(1, repeat_factor, 1)
            else:
                expanded_orig = orig_cond

            # Expand new_cond by repeating to LCM length
            if target_seq_len < lcm_len:
                repeat_factor = lcm_len // target_seq_len
                expanded_new = new_cond.repeat(1, repeat_factor, 1)
            else:
                expanded_new = new_cond

            modified_chunks = []
            for i, ct in enumerate(cond_or_uncond):
                if ct == 0:  # Positive: use swapped conditioning
                    modified_chunks.append(expanded_new[0:1])
                else:  # Negative/Uncond: use original conditioning
                    modified_chunks.append(expanded_orig[i : i + 1])

            modified_cond = torch.cat(modified_chunks, dim=0)

        # Cache the result
        if len(self._swap_cache) > 100:
            self._swap_cache.clear()
        self._swap_cache[cache_key] = modified_cond

        # Finalize
        c = dict(c)
        c["c_crossattn"] = modified_cond

        if self.debug and not self._first_swap_logged:
            logger.info(f"[A1111 Hook] Swapped cond for step {step_idx}")
            self._first_swap_logged = True

        self._last_logged_step = step_idx

        if existing_wrapper is not None:
            return existing_wrapper(
                apply_model_func,
                {
                    "input": input_x,
                    "timestep": timestep,
                    "c": c,
                    "cond_or_uncond": cond_or_uncond,
                },
            )
        return apply_model_func(input_x, timestep, **c)

    def clone(self):
        """Clone this hook for use in different conditioning contexts."""
        c = super().clone()
        c.step_embeddings = self.step_embeddings
        c.default_steps = self.default_steps
        c.debug = self.debug
        c._last_logged_step = self._last_logged_step
        # Share the cache dictionary to ensure 2nd order samplers enjoy performance gains
        c._swap_cache = self._swap_cache
        return c


def create_step_schedule_cond(
    step_embeddings, default_steps=28, base_cond=None, base_pooled=None, debug=False
):
    """
    Create conditioning with step schedule hook attached.

    This version uses TransformerOptionsHook attached to conditioning,
    eliminating the need for MODEL input. The hook automatically receives
    sample_sigmas during sampling and calculates actual step count from it.

    Args:
        step_embeddings: List of (cond, pooled) per step
        default_steps: Default step count used for parsing (for scaling)
        base_cond: Base conditioning tensor
        base_pooled: Base pooled output
        debug: Whether to log verbose scheduling information

    Returns:
        Conditioning list with hook attached
    """
    # Create the hook
    hook = A1111StepConditioningHook(step_embeddings, default_steps, debug=debug)
    hook_group = HookGroup()
    hook_group.add(hook)

    # Create base conditioning
    cond_dict = {
        "pooled_output": base_pooled,
        "a1111_step_schedule": {
            "embeddings": step_embeddings,
            "default_steps": default_steps,
        },
    }
    conditioning = [[base_cond, cond_dict]]

    # Attach hook to conditioning
    conditioning = set_hooks_for_conditioning(
        conditioning, hooks=hook_group, append_hooks=True
    )

    return conditioning
