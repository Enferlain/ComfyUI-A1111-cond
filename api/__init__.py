"""
A1111 Prompt API Package

Contains REST API endpoints for the frontend.
"""

# Import autocomplete functionality (no server dependencies)
from .autocomplete import get_database, ensure_database_loaded, TagDatabase

# Conditionally import tokenize functionality (requires server)
try:
    from .tokenize import tokenize_prompt, get_tokenizer

    _HAS_TOKENIZE = True
except ImportError:
    _HAS_TOKENIZE = False
    tokenize_prompt = None
    get_tokenizer = None

__all__ = [
    "get_database",
    "ensure_database_loaded",
    "TagDatabase",
]

if _HAS_TOKENIZE:
    __all__.extend(["tokenize_prompt", "get_tokenizer"])
