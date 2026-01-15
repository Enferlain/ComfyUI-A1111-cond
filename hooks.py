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

import torch
import logging


class StepConditioningHandler:
    """
    Handles step-based conditioning switching.

    Stores the per-step embeddings and provides the wrapper function
    that swaps conditioning based on current sigma/timestep.
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

        # Determine current step
        step_idx = self.get_step_from_sigma(sigma_val, sample_sigmas)
        step_idx = max(0, min(step_idx, len(self.step_embeddings) - 1))

        # Get target conditioning for this step
        target_cond, target_pooled = self.step_embeddings[step_idx]

        # Log step transitions
        if self._last_logged_step != step_idx:
            logging.debug(f"A1111 Step: {step_idx}/{self.steps}, sigma={sigma_val:.4f}")
            self._last_logged_step = step_idx

        # Swap conditioning if we have a target
        if target_cond is not None and "c_crossattn" in c:
            # Get original conditioning info
            orig_cond = c["c_crossattn"]
            device = orig_cond.device
            dtype = orig_cond.dtype

            # Prepare swapped conditioning
            new_cond = target_cond.to(device=device, dtype=dtype)

            # Handle batch size
            if new_cond.shape[0] != orig_cond.shape[0]:
                B = orig_cond.shape[0]
                pattern = [B] + [1] * (new_cond.ndim - 1)
                new_cond = new_cond.repeat(*pattern)

            # Create modified c dict
            c = dict(c)
            c["c_crossattn"] = new_cond

            # Also update pooled if available and SDXL
            if target_pooled is not None and "y" in c:
                new_pooled = target_pooled.to(device=device, dtype=dtype)
                if new_pooled.shape[0] != c["y"].shape[0]:
                    B = c["y"].shape[0]
                    pattern = [B] + [1] * (new_pooled.ndim - 1)
                    new_pooled = new_pooled.repeat(*pattern)
                c["y"] = new_pooled

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
