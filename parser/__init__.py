"""
A1111 Prompt Parser Package

Contains the Lark grammar parser and scheduling logic for A1111-style prompts.
"""

from .wildcards import (
    expand_wildcards,
    expand_wildcards_for_token_count,
    get_wildcard_options,
    list_available_wildcards,
)

try:
    from .grammar import GRAMMAR, get_parser, reset_parser
    from .scheduler import get_prompt_schedule
except ImportError as exc:
    _PARSER_IMPORT_ERROR = exc
    GRAMMAR = None

    def get_parser():
        raise _PARSER_IMPORT_ERROR

    def reset_parser():
        return None

    def get_prompt_schedule(*args, **kwargs):
        raise _PARSER_IMPORT_ERROR

__all__ = [
    "GRAMMAR",
    "get_parser",
    "reset_parser",
    "get_prompt_schedule",
    "expand_wildcards",
    "expand_wildcards_for_token_count",
    "get_wildcard_options",
    "list_available_wildcards",
]
