"""
A1111 Prompt Nodes Package

Contains the main positive and negative prompt nodes.
"""

from .prompt_node import A1111PromptNode
from .negative_node import A1111PromptNegative

NODE_CLASS_MAPPINGS = {
    "A1111Prompt": A1111PromptNode,
    "A1111PromptNegative": A1111PromptNegative,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "A1111Prompt": "A1111 Style Prompt",
    "A1111PromptNegative": "A1111 Style Prompt (Negative)",
}

__all__ = [
    "A1111PromptNode",
    "A1111PromptNegative",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
