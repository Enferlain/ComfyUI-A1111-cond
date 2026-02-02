import unittest
import sys
from pathlib import Path
import torch
from unittest.mock import MagicMock

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

# Mock ComfyUI dependencies
from unittest.mock import MagicMock

mock_hooks = MagicMock()


class DummyHook:
    def __init__(self, *args, **kwargs):
        pass

    def clone(self):
        return self


mock_hooks.TransformerOptionsHook = DummyHook
mock_hooks.EnumHookScope = MagicMock()

sys.modules["comfy.hooks"] = mock_hooks
sys.modules["server"] = MagicMock()


from hooks import A1111StepConditioningHook


class TestHooks(unittest.TestCase):
    def test_get_step_from_sigma(self):
        # mock sample_sigmas [10.0, 8.0, 5.0, 0.0]
        # step 0: 10.0 -> 8.0
        # step 1: 8.0 -> 5.0
        # step 2: 5.0 -> 0.0
        sigmas = torch.tensor([10.0, 8.0, 5.0, 0.0])
        hook = A1111StepConditioningHook(step_embeddings=[])

        # Exact match start
        self.assertEqual(hook.get_step_from_sigma(10.0, sigmas), 0)
        # Inside first step
        self.assertEqual(hook.get_step_from_sigma(9.0, sigmas), 0)
        # Boundary: 8.0 is the START of step 1
        self.assertEqual(hook.get_step_from_sigma(8.0, sigmas), 1)
        # Inside second step
        self.assertEqual(hook.get_step_from_sigma(7.0, sigmas), 1)
        # Exact match end of last step: 0.0 is the start of the "final" interval
        # (Though samplers usually stop at 0, this returns len-2 as per hooks.py)
        self.assertEqual(hook.get_step_from_sigma(0.0, sigmas), 2)

    def test_step_index_scaling(self):
        # sampler has 10 steps, we have 20 embeddings
        sigmas = torch.linspace(10, 0, 11)  # length 11 -> 10 steps
        embeddings = [i for i in range(21)]  # length 21 -> step 0 to 20
        hook = A1111StepConditioningHook(step_embeddings=embeddings)

        # Mock the wrapper logic part that calculates step_idx
        def get_scaled_step(sigma_val):
            raw_idx = hook.get_step_from_sigma(sigma_val, sigmas)
            actual_steps = len(sigmas) - 1
            emb_steps = len(embeddings) - 1
            return round(raw_idx * emb_steps / actual_steps)

        # Start (step 0 of 10) -> step 0 of 20
        self.assertEqual(get_scaled_step(10.0), 0)
        # Middle (step 5 of 10) -> step 10 of 20
        self.assertEqual(get_scaled_step(5.0), 10)
        # End (step 9 of 10) -> step 18 of 20 (it's N-1 for last index usually)
        # actually last sigma is step 9.
        self.assertEqual(get_scaled_step(sigmas[9].item()), 18)

    def test_model_function_wrapper_swaps(self):
        # Prepare 3 embeddings (0, 1, 2)
        emb0 = torch.ones((1, 1, 4)) * 0.0
        emb1 = torch.ones((1, 1, 4)) * 1.0
        emb2 = torch.ones((1, 1, 4)) * 2.0
        hook = A1111StepConditioningHook(
            step_embeddings=[(emb0, None), (emb1, None), (emb2, None)]
        )
        hook.debug = True

        # Mock arguments
        apply_model_func = MagicMock(return_value="model_output")
        input_x = torch.zeros((1, 4))
        timestep = torch.tensor([10.0])
        sigmas = torch.tensor([10.0, 5.0, 0.0])  # 2 steps: [10->5, 5->0]

        c = {
            "c_crossattn": torch.ones((2, 1, 4)) * -1.0,  # original (batch=2)
            "transformer_options": {"sample_sigmas": sigmas},
        }
        cond_or_uncond = [0, 1]  # 0 = positive (swap), 1 = negative (keep)

        args = {
            "input": input_x,
            "timestep": timestep,
            "c": c,
            "cond_or_uncond": cond_or_uncond,
        }

        # Step 0 (sigma 10.0) -> raw_idx 0
        # Scaling: round(0 * 2 / 2) = 0 -> emb0
        hook.model_function_wrapper(apply_model_func, args)

        call_args = apply_model_func.call_args[1]
        new_cond = call_args["c_crossattn"]
        self.assertTrue(torch.allclose(new_cond[0], emb0[0]))

        # Step 1 (sigma 5.0) -> raw_idx 1
        # Scaling: round(1 * 2 / 2) = 1 -> emb1
        timestep = torch.tensor([5.0])
        args["timestep"] = timestep
        hook.model_function_wrapper(apply_model_func, args)

        call_args = apply_model_func.call_args[1]
        new_cond = call_args["c_crossattn"]
        self.assertTrue(torch.allclose(new_cond[0], emb1[0]))


if __name__ == "__main__":
    unittest.main()
