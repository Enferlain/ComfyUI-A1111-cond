"""
A1111 Prompt Grammar

Lark grammar definition for A1111-style prompt syntax.

Supported Syntax:
- Emphasis: (text:1.2), (text), [text]
- Scheduling: [from:to:when], [add:when], [remove::when]
- Alternation: [A|B|C]
- BREAK: Splits prompt into isolated chunks

Step vs Percentage:
- If 'when' has a decimal point (e.g., 0.5), it's a percentage
- If 'when' is an integer (e.g., 10), it's a step count (requires steps input)
"""

import lark


# A1111-style Lark grammar with scheduled alternation extension
# Extended to support [a|b:when] and [a|b::when] syntax
GRAMMAR = r"""
!start: (prompt | /[][():]/+)*
prompt: (emphasized | scheduled | alternate_scheduled | alternate | plain | WHITESPACE)*
!emphasized: "(" prompt ")"
        | "(" prompt ":" prompt ")"
        | "[" prompt "]"
scheduled: "[" [prompt ":"] prompt ":" [WHITESPACE] NUMBER [WHITESPACE] "]"
alternate_scheduled: "[" prompt ("|" [prompt])+ ":" DOUBLE_COLON? [WHITESPACE] NUMBER [WHITESPACE] "]"
alternate: "[" prompt ("|" [prompt])+ "]"
DOUBLE_COLON: ":"
WHITESPACE: /\s+/
plain: /([^\\[\]():|]|\\.)+/
%import common.SIGNED_NUMBER -> NUMBER
"""

# Cached parser instance
_parser = None


def get_parser():
    """Get or create cached Lark parser."""
    global _parser
    if _parser is None:
        _parser = lark.Lark(GRAMMAR)
    return _parser


def reset_parser():
    """Reset the cached parser. Call this after grammar changes."""
    global _parser
    _parser = None
