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

    def __init__(self, step_embeddings, default_steps=20):
        """
        Args:
            step_embeddings: List of (cond_tensor, pooled_tensor) per step index
            default_steps: Default step count used for parsing (for scaling)
        """
        super().__init__(hook_scope=EnumHookScope.AllConditioning)
        self.step_embeddings = step_embeddings
        self.default_steps = default_steps
        self._last_logged_step = -1
        self._first_swap_logged = False  # Track if we've logged the first swap

        # Don't use transformers_dict - we'll set model_function_wrapper directly
        self.transformers_dict = {}

    def add_hook_patches(self, model, model_options, target_dict, registered):
        """Override to set model_function_wrapper directly on model_options."""
        if not self.should_register(model, model_options, target_dict, registered):
            return False

        # Check if there's already a wrapper - we need to chain them
        existing_wrapper = model_options.get("model_function_wrapper")
        if existing_wrapper is not None:
            logger.warning(
                f"[A1111 Hook] Found existing model_function_wrapper, will chain them"
            )

            # Create a chained wrapper
            def chained_wrapper(apply_model_func, args):
                # Call our wrapper first, which will call the existing one
                return self.model_function_wrapper(
                    apply_model_func, args, existing_wrapper
                )

            model_options["model_function_wrapper"] = chained_wrapper
        else:
            logger.info(
                f"[A1111 Hook] Registering model_function_wrapper on model_options"
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
        # Log on very first call to confirm wrapper is active
        if self._last_logged_step == -1:
            logger.info(
                f"[A1111 Hook] ========== WRAPPER CALLED - HOOK IS ACTIVE =========="
            )
            if existing_wrapper is not None:
                logger.info(f"[A1111 Hook] Chaining with existing wrapper")

        input_x = args["input"]
        timestep = args["timestep"]
        c = args["c"]
        cond_or_uncond = args["cond_or_uncond"]

        # Get sigma from timestep
        if isinstance(timestep, torch.Tensor):
            sigma_val = timestep[0].item() if timestep.ndim > 0 else timestep.item()
        else:
            sigma_val = float(timestep)

        # Get sample_sigmas from transformer_options in c
        transformer_options = c.get("transformer_options", {})
        sample_sigmas = transformer_options.get("sample_sigmas")

        if sample_sigmas is None:
            # No sigmas available - shouldn't happen, but fallback to first step
            logger.warning("[A1111 Hook] No sample_sigmas found, using first step")
            if existing_wrapper is not None:
                return existing_wrapper(apply_model_func, args)
            return apply_model_func(input_x, timestep, **c)

        # Calculate actual total steps from sigmas
        actual_steps = len(sample_sigmas) - 1

        # Log on first call
        if self._last_logged_step == -1:
            logger.info(f"[A1111 Hook] Actual sampler steps: {actual_steps}")
            logger.info(
                f"[A1111 Hook] Embeddings prepared for: {len(self.step_embeddings)} steps"
            )
            logger.info(f"[A1111 Hook] Default parse steps: {self.default_steps}")

        # Determine current step from sigma
        step_idx = self.get_step_from_sigma(sigma_val, sample_sigmas)

        # Clamp step_idx to valid range (same as old StepConditioningHandler)
        step_idx = max(0, min(step_idx, len(self.step_embeddings) - 1))

        # Get target conditioning for this step
        target_cond, target_pooled = self.step_embeddings[step_idx]

        # Log step changes (deduplicated)
        if step_idx != self._last_logged_step:
            logger.debug(
                f"[A1111 Hook] Step {step_idx}/{actual_steps - 1} (sigma={sigma_val:.4f})"
            )
            self._last_logged_step = step_idx

        # IMPORTANT: The batch contains both positive and negative conditioning.
        # cond_or_uncond tells us which is which: 0=positive, 1=negative
        # We only swap the POSITIVE conditioning, keep negative as-is for CFG to work.
        #
        # When positive and negative have very different sequence lengths (e.g., 385 vs 77),
        # ComfyUI can't batch them together and processes them separately. Skip the swap
        # entirely when processing negative-only batches.
        has_positive = any(ct == 0 for ct in cond_or_uncond)

        # Debug logging on first call - check conditions
        if not self._first_swap_logged:
            logger.info(f"[A1111 Hook] === CHECKING SWAP CONDITIONS ===")
            logger.info(
                f"[A1111 Hook] target_cond is not None: {target_cond is not None}"
            )
            logger.info(f"[A1111 Hook] 'c_crossattn' in c: {'c_crossattn' in c}")
            logger.info(f"[A1111 Hook] has_positive: {has_positive}")
            logger.info(f"[A1111 Hook] cond_or_uncond: {cond_or_uncond}")
            logger.info(f"[A1111 Hook] Keys in c: {list(c.keys())}")

        if target_cond is not None and "c_crossattn" in c and has_positive:
            orig_cond = c["c_crossattn"]
            device = orig_cond.device
            dtype = orig_cond.dtype

            # Log details on first swap
            if not self._first_swap_logged:
                logger.info(f"[A1111 Hook] === FIRST CONDITIONING SWAP ===")
                logger.info(
                    f"[A1111 Hook] Original c_crossattn shape: {orig_cond.shape}"
                )
                logger.info(f"[A1111 Hook] Target cond shape: {target_cond.shape}")
                logger.info(f"[A1111 Hook] cond_or_uncond: {cond_or_uncond}")
                logger.info(f"[A1111 Hook] Batch size: {len(cond_or_uncond)}")
                # Check if they're actually different
                are_same = torch.allclose(orig_cond, target_cond, rtol=1e-5, atol=1e-8)
                logger.info(
                    f"[A1111 Hook] Original and target are identical: {are_same}"
                )
                if are_same:
                    logger.warning(
                        f"[A1111 Hook] WARNING: Original and target conditioning are the same! No actual swap needed."
                    )

            # Prepare swapped conditioning
            new_cond = target_cond.to(device=device, dtype=dtype).clone()

            # Handle sequence length mismatch using LCM-based repeat padding
            target_seq_len = new_cond.shape[1]
            orig_seq_len = orig_cond.shape[1]

            if target_seq_len != orig_seq_len:
                # Use LCM like ldm_patched does for proper batching
                lcm_len = math.lcm(target_seq_len, orig_seq_len)

                if not self._first_swap_logged:
                    logger.info(
                        f"[A1111 Hook] Sequence length mismatch: target={target_seq_len}, orig={orig_seq_len}, LCM={lcm_len}"
                    )

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

                # Create output tensor starting with expanded original
                modified_cond = expanded_orig.clone()

                # Swap only positive conditioning positions with our step embedding
                swapped_count = 0
                for batch_idx, cond_type in enumerate(cond_or_uncond):
                    if cond_type == 0:  # Positive conditioning
                        modified_cond[batch_idx] = expanded_new[0]
                        swapped_count += 1

                if not self._first_swap_logged:
                    logger.info(
                        f"[A1111 Hook] Swapped {swapped_count} positive conditioning(s)"
                    )
            else:
                # Lengths match - simple case
                modified_cond = orig_cond.clone()
                swapped_count = 0
                for batch_idx, cond_type in enumerate(cond_or_uncond):
                    if cond_type == 0:
                        modified_cond[batch_idx] = new_cond[0]
                        swapped_count += 1

                if not self._first_swap_logged:
                    logger.info(
                        f"[A1111 Hook] Swapped {swapped_count} positive conditioning(s) (no length mismatch)"
                    )

            # Create modified c dict with swapped conditioning
            c = dict(c)
            c["c_crossattn"] = modified_cond

            if not self._first_swap_logged:
                logger.info(
                    f"[A1111 Hook] Final modified_cond shape: {modified_cond.shape}"
                )
                logger.info(f"[A1111 Hook] === SWAP COMPLETE ===")
                self._first_swap_logged = True  # Mark first swap as complete
        elif not self._first_swap_logged:
            logger.warning(f"[A1111 Hook] Conditioning swap skipped!")
            if target_cond is None:
                logger.warning(f"[A1111 Hook]   - target_cond is None")
            if "c_crossattn" not in c:
                logger.warning(f"[A1111 Hook]   - c_crossattn not in c")
            if not has_positive:
                logger.warning(f"[A1111 Hook]   - no positive conditioning in batch")

            # NOTE: Pooled output (y) swapping disabled - testing if it improves A1111 parity
            # The pooled output might have different timing characteristics that affect
            # how the style "settles" in later steps.
            # if target_pooled is not None and "y" in c:
            #     orig_y = c["y"]
            #     if orig_y.shape[-1] == 2816:  # SDXL
            #         new_pooled = target_pooled.to(device=device, dtype=dtype).clone()
            #         modified_y = orig_y.clone()
            #
            #         # Only swap positive positions
            #         for batch_idx, cond_type in enumerate(cond_or_uncond):
            #             if cond_type == 0 and batch_idx < modified_y.shape[0]:
            #                 # Replace first 1280 dims with our pooled output
            #                 modified_y[batch_idx, :1280] = new_pooled[0]
            #
            #         c["y"] = modified_y
            #         logging.debug(f"  Also swapped pooled output (first 1280 of y)")

        # CRITICAL: Update args with modified c before calling model
        # This ensures our conditioning changes are actually used
        args = dict(args)
        args["c"] = c

        # Call the actual model (or existing wrapper if chained)
        if existing_wrapper is not None:
            return existing_wrapper(apply_model_func, args)
        return apply_model_func(input_x, timestep, **c)

    def clone(self):
        """Clone this hook for use in different conditioning contexts."""
        c = super().clone()
        c.step_embeddings = self.step_embeddings
        c.default_steps = self.default_steps
        c._last_logged_step = self._last_logged_step
        return c


def create_step_schedule_cond(step_embeddings, default_steps, base_cond, base_pooled):
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

    Returns:
        Conditioning list with hook attached
    """
    # Create the hook
    hook = A1111StepConditioningHook(step_embeddings, default_steps)
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
