"""
A1111 Step-Based Conditioning

Implements true A1111-style step-based prompt scheduling and alternation.

This module provides:
1. StepConditioningHandler - stores per-step embeddings
2. setup_step_conditioning_on_model - registers the wrapper on ModelPatcher
3. Helper functions for creating step-scheduled conditioning

The key insight is that the `model_function_wrapper` in model_options receives
the timestep and conditioning, allowing us to swap conditioning per-step.
"""

import math
import torch


class StepConditioningHandler:
    """
    Handles step-based conditioning switching.

    Stores the per-step embeddings and provides the wrapper function
    that swaps conditioning based on current step count (like A1111).
    """

    def __init__(self, step_embeddings, steps, sample_sigmas=None):
        """
        Args:
            step_embeddings: List of (cond_tensor, pooled_tensor) per step index
            steps: Total user-specified steps
            sample_sigmas: Will be set during sampling from transformer_options
        """
        self.step_embeddings = step_embeddings
        self.steps = steps
        self.call_count = 0  # A1111-style step counter
        self.sample_sigmas = sample_sigmas
        self._last_logged_step = -1

    def get_step_from_sigma(self, sigma_val, sample_sigmas=None):
        """
        Determine which step index we're at based on current sigma.
        """
        sigmas = sample_sigmas if sample_sigmas is not None else self.sample_sigmas

        if sigmas is None or len(sigmas) == 0:
            return 0

        num_sigmas = len(sigmas)

        for i in range(num_sigmas - 1):
            s_start = (
                sigmas[i].item() if isinstance(sigmas[i], torch.Tensor) else sigmas[i]
            )
            s_end = (
                sigmas[i + 1].item()
                if isinstance(sigmas[i + 1], torch.Tensor)
                else sigmas[i + 1]
            )

            if s_start >= sigma_val > s_end:
                return i

            if abs(s_start - sigma_val) < 1e-4:
                return i

        last_sig = (
            sigmas[-1].item() if isinstance(sigmas[-1], torch.Tensor) else sigmas[-1]
        )
        if sigma_val <= last_sig + 1e-4:
            return num_sigmas - 2

        return 0

    def model_function_wrapper(self, apply_model_func, args):
        """
        Wrapper function that intercepts model application and swaps conditioning.

        This is called by ComfyUI's sampling code when model_function_wrapper
        is set in model_options.
        """
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
        sample_sigmas = transformer_options.get("sample_sigmas", self.sample_sigmas)

        # Determine current step from sigma
        step_idx = self.get_step_from_sigma(sigma_val, sample_sigmas)
        step_idx = max(0, min(step_idx, len(self.step_embeddings) - 1))

        # Get target conditioning for this step
        target_cond, target_pooled = self.step_embeddings[step_idx]

        # Track step for deduplication (debug logging removed for cleaner output)
        self._last_logged_step = step_idx

        # IMPORTANT: The batch contains both positive and negative conditioning.
        # cond_or_uncond tells us which is which: 0=positive, 1=negative
        # We only swap the POSITIVE conditioning, keep negative as-is for CFG to work.
        #
        # When positive and negative have very different sequence lengths (e.g., 385 vs 77),
        # ComfyUI can't batch them together and processes them separately. Skip the swap
        # entirely when processing negative-only batches.
        has_positive = any(ct == 0 for ct in cond_or_uncond)
        if target_cond is not None and "c_crossattn" in c and has_positive:
            orig_cond = c["c_crossattn"]
            device = orig_cond.device
            dtype = orig_cond.dtype

            # Prepare swapped conditioning
            new_cond = target_cond.to(device=device, dtype=dtype).clone()

            # Handle sequence length mismatch using LCM-based repeat padding
            # This matches ldm_patched/ComfyUI's CONDCrossAttn.concat() behavior
            # and A1111's default handling (padding with repeat doesn't change result)
            target_seq_len = new_cond.shape[1]
            orig_seq_len = orig_cond.shape[1]

            if target_seq_len != orig_seq_len:
                # Use LCM like ldm_patched does for proper batching
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

                # Create output tensor starting with expanded original
                modified_cond = expanded_orig.clone()

                # Swap only positive conditioning positions with our step embedding
                for batch_idx, cond_type in enumerate(cond_or_uncond):
                    if cond_type == 0:  # Positive conditioning
                        modified_cond[batch_idx] = expanded_new[0]
            else:
                # Lengths match - simple case
                modified_cond = orig_cond.clone()
                for batch_idx, cond_type in enumerate(cond_or_uncond):
                    if cond_type == 0:
                        modified_cond[batch_idx] = new_cond[0]

            # Create modified c dict with swapped conditioning
            c = dict(c)
            c["c_crossattn"] = modified_cond

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

        # Call the actual model
        return apply_model_func(input_x, timestep, **c)


def setup_step_conditioning_on_model(model_patcher, step_embeddings, steps):
    """
    Set up step-based conditioning on a ModelPatcher.

    This should be called from a node that has MODEL input to register
    the wrapper. The wrapper will swap conditioning per-step during sampling.

    Args:
        model_patcher: The ModelPatcher to configure
        step_embeddings: List of (cond, pooled) per step
        steps: Total steps

    Returns:
        The modified model_patcher (same object, modified in place)
    """
    handler = StepConditioningHandler(step_embeddings, steps)
    model_patcher.model_options["model_function_wrapper"] = (
        handler.model_function_wrapper
    )
    return model_patcher


def create_step_schedule_cond(step_embeddings, steps, base_cond, base_pooled):
    """
    Create a conditioning tuple with step schedule metadata attached.

    The metadata allows other nodes to access the step schedule if needed.
    The actual swapping is done by the model_function_wrapper.

    Args:
        step_embeddings: List of (cond, pooled) per step
        steps: Total user-specified steps
        base_cond: Base conditioning tensor
        base_pooled: Base pooled output

    Returns:
        Conditioning list (to be wrapped in tuple by caller)
    """
    cond_dict = {
        "pooled_output": base_pooled,
        "a1111_step_schedule": {
            "embeddings": step_embeddings,
            "steps": steps,
        },
    }

    return [[base_cond, cond_dict]]
