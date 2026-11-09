"""Contract between prompt orchestration and a conditioning implementation."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class EncoderCapabilities:
    breaks: bool
    scheduling: bool
    normalization: bool


class PromptEncoder(ABC):
    """Return (conditioning tensor, complete metadata) without retaining state.

    Implementations own tokenization, weighting, segmentation and the metadata
    required by the consuming model. Nodes own parsing, caching and schedules.
    """

    name: str
    capabilities: EncoderCapabilities

    @abstractmethod
    def matches(self, clip, streams):
        """Whether the encoder/model contract supports these token streams."""

    def validate_request(self, text="", *, normalization=False, scheduled=False):
        if re.search(r"\bBREAK\b", text) and not self.capabilities.breaks:
            raise ValueError(f"{self.name} BREAK is not supported. Use an unsplit prompt.")
        if scheduled and not self.capabilities.scheduling:
            raise ValueError(
                f"{self.name} scheduling/alternation is not supported. Use a static prompt."
            )
        if normalization and not self.capabilities.normalization:
            raise ValueError(
                f"{self.name} normalization is not supported. Disable normalization."
            )

    def validate_schedule_metadata(self, metadata):
        # The shared hook swaps the embedding tensor only. pooled_output is
        # permitted for compatibility but stays pinned to the first prompt.
        if set(metadata) - {"pooled_output"}:
            raise ValueError(
                "Prompt scheduling cannot swap this encoder's auxiliary "
                "conditioning fields yet. Use a static prompt."
            )

    @abstractmethod
    def encode(self, clip, text, normalization=False, debug=False):
        """Encode text after validating supported operations; preserve metadata."""
