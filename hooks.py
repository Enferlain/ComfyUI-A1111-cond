"""
Hooks for advanced prompting features like alternation [A|B]

NOTE: This module is currently a PLACEHOLDER.
The AlternatingHook is not yet integrated into the sampling loop.
True per-step alternation requires custom sampler patching.

Current implementation uses blending (averaging conditionings) as a workaround.

TODO: Integrate with ComfyUI's sampler hooks when API is available.
"""


class AlternatingHook:
    """
    Hook that alternates between two conditionings based on timestep.
    For [A|B] syntax: odd steps use A, even steps use B.

    WARNING: Not currently wired into sampling. See nodes.py for blend workaround.
    """

    def __init__(self, options):
        """
        Args:
            options: List of WeightedPart objects representing the alternating choices
        """
        self.options = options
        self.current_index = 0

    def __call__(self, model, step, total_steps, **kwargs):
        """
        Called during sampling to determine which conditioning to use.

        Returns:
            dict with strength multiplier for this conditioning
        """
        # Simple alternation: switch every step
        # Step is 0-indexed, so even steps = option 0, odd steps = option 1
        index = step % len(self.options)

        # Return strength 1.0 for active option, 0.0 for others
        # But we're only supporting 2 options for now (A|B)
        if index == self.current_index:
            return {"strength": 1.0}
        else:
            return {"strength": 0.0}
