"""
A1111 Prompt Node for ComfyUI

A custom ComfyUI node that implements A1111-style prompt handling
with proper isolation, emphasis math, scheduling, and alternation.

Package Structure:
    nodes/        - ComfyUI node definitions (positive & negative)
    parser/       - Lark grammar and scheduling logic
    api/          - REST API endpoints for frontend
    js/           - Frontend JavaScript extensions
    data/         - Tag databases and wildcards
    hooks.py      - ComfyUI model hooks for step-based conditioning
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from . import api  # Register tokenization API endpoint

# Tell ComfyUI to load JS files from the "js" folder
WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
