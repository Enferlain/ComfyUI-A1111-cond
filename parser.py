"""
A1111 Prompt Parser (Recursive)

Uses Lark grammar parser like A1111 for proper nested syntax support.

Supported Syntax:
- Emphasis: (text:1.2), (text), [text]
- Scheduling: [from:to:when], [add:when], [remove::when]
- Alternation: [A|B|C]
- BREAK: Splits prompt into isolated chunks

Step vs Percentage:
- If 'when' has a decimal point (e.g., 0.5), it's a percentage
- If 'when' is an integer (e.g., 10), it's a step count (requires steps input)
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

# Try to import lark, fall back to simple parsing if not available
try:
    import lark

    HAS_LARK = True
except ImportError:
    HAS_LARK = False


@dataclass
class WeightedPart:
    """Represents a parsed prompt segment with weight and timing info."""

    text: str
    weight: float = 1.0
    start_percent: float = 0.0
    end_percent: float = 1.0
    is_alternating: bool = False
    alternating_options: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        """Return the text content for string conversion."""
        return self.text


# A1111-style Lark grammar
GRAMMAR = r"""
!start: (prompt | /[][():]/+)*
prompt: (emphasized | scheduled | alternate | plain | WHITESPACE)*
!emphasized: "(" prompt ")"
        | "(" prompt ":" prompt ")"
        | "[" prompt "]"
scheduled: "[" [prompt ":"] prompt ":" [WHITESPACE] NUMBER [WHITESPACE] "]"
alternate: "[" prompt ("|" [prompt])+ "]"
WHITESPACE: /\s+/
plain: /([^\\\\[\]():|]|\\.)+/
%import common.SIGNED_NUMBER -> NUMBER
"""


def parse_a1111_prompt(
    prompt: str, steps: Optional[int] = None
) -> List[List[WeightedPart]]:
    """
    Main entry point.

    Args:
        prompt: The prompt string to parse
        steps: Total sampling steps (needed for step-based scheduling)

    Returns: List[List[WeightedPart]]
        Outer list = BREAK chunks
        Inner list = Parts with timing/weight info
    """
    # Split by BREAK first
    break_chunks = re.split(r"\s*\bBREAK\b\s*", prompt)

    if HAS_LARK:
        return [_parse_chunk_lark(chunk, steps) for chunk in break_chunks]
    else:
        return [_parse_chunk_simple(chunk, steps) for chunk in break_chunks]


def get_prompt_schedule(prompt: str, steps: int) -> List[Tuple[int, str]]:
    """
    A1111-style prompt schedule generation.

    Returns list of (end_at_step, prompt_text) tuples representing
    when each prompt variant should be used.

    This properly handles nested alternation and scheduling.
    """
    if not HAS_LARK:
        # Fallback for simple prompts
        return [(steps, prompt)]

    # Split by BREAK first, process each chunk, then rejoin
    break_chunks = re.split(r"\s*\bBREAK\b\s*", prompt)

    try:
        parser = lark.Lark(GRAMMAR)
        trees = [parser.parse(chunk) for chunk in break_chunks]
    except lark.exceptions.LarkError:
        return [(steps, prompt)]

    # Collect all relevant steps from all chunks
    all_steps = _collect_steps(steps, trees)

    # Generate prompt text for each step
    schedule = []
    for step in all_steps:
        chunk_texts = []
        for tree in trees:
            text = _at_step(step, steps, tree)
            if text.strip():
                chunk_texts.append(text)
        full_prompt = " BREAK ".join(chunk_texts)
        schedule.append((step, full_prompt))

    return schedule


def _collect_steps(steps: int, trees) -> List[int]:
    """Collect all steps where the prompt changes."""
    result = {steps}  # Always include final step

    class CollectSteps(lark.Visitor):
        def scheduled(self, tree):
            s = str(tree.children[-2])
            v = float(s)
            if "." in s:
                # Percentage
                step_num = int(v * steps)
            else:
                # Step count
                step_num = int(v)
            step_num = max(1, min(steps, step_num))
            if step_num >= 1:
                result.add(step_num)

        def alternate(self, tree):
            # Alternation changes every step
            result.update(range(1, steps + 1))

    for tree in trees:
        CollectSteps().visit(tree)

    return sorted(result)


def _at_step(step: int, total_steps: int, tree) -> str:
    """
    Generate the prompt text for a specific step.

    This follows A1111's approach of walking the tree and picking
    the correct option at each node based on the current step.
    """

    class AtStep(lark.Transformer):
        def scheduled(self, args):
            # A1111 pattern: before, after, _, when, _ = args
            # Use positional access like A1111
            before = args[0] if len(args) > 0 else None
            after = args[1] if len(args) > 1 else None
            when_token = args[3] if len(args) > 3 else None

            # Fallback: find NUMBER token if not at expected position
            if when_token is None or not (
                isinstance(when_token, lark.Token) and when_token.type == "NUMBER"
            ):
                for arg in args:
                    if isinstance(arg, lark.Token) and arg.type == "NUMBER":
                        when_token = arg
                        break

            if when_token is None:
                for a in args:
                    if a is not None:
                        yield a
                return

            when_str = str(when_token)
            if "." in when_str:
                when_step = int(float(when_str) * total_steps)
            else:
                when_step = int(float(when_str))

            # A1111: yield before if step <= when, else yield after
            if step <= when_step:
                if before is not None:
                    yield before
            else:
                if after is not None:
                    yield after

        def alternate(self, args):
            # Filter out empty/None
            options = [a for a in args if a]
            if not options:
                yield ""
                return
            idx = (step - 1) % len(options)
            yield options[idx]

        def emphasized(self, args):
            # Keep emphasis syntax in output
            if len(args) == 1:
                for a in args:
                    yield a
            elif len(args) >= 2:
                # (text:weight) - preserve weight
                yield "("
                yield from self._flatten_arg(args[0])
                yield ":"
                yield from self._flatten_arg(args[1])
                yield ")"
            else:
                for a in args:
                    yield a

        def _flatten_arg(self, arg):
            if isinstance(arg, str):
                yield arg
            elif isinstance(arg, lark.Token):
                yield str(arg)
            elif hasattr(arg, "__iter__"):
                for item in arg:
                    yield from self._flatten_arg(item)
            elif arg is not None:
                yield str(arg)

        def plain(self, args):
            yield args[0].value if args else ""

        def start(self, args):
            def flatten(x):
                if isinstance(x, str):
                    yield x
                elif isinstance(x, lark.Token):
                    yield str(x)
                elif x is None or x == ():
                    pass
                elif hasattr(x, "__iter__"):
                    for item in x:
                        yield from flatten(item)
                else:
                    yield str(x)

            return "".join(flatten(args))

        def __default__(self, data, children, meta):
            for child in children:
                yield child

    return AtStep().transform(tree)


def _parse_chunk_lark(chunk: str, steps: Optional[int]) -> List[WeightedPart]:
    """Parse a single chunk using Lark grammar."""
    try:
        parser = lark.Lark(GRAMMAR)
        tree = parser.parse(chunk)
        return _extract_parts_from_tree(tree, steps)
    except lark.exceptions.LarkError:
        # Fallback to simple parsing on error
        return _parse_chunk_simple(chunk, steps)


def _convert_when_to_percent(when_str: str, steps: Optional[int]) -> float:
    """
    Convert a 'when' value to a percentage.
    - If has decimal point: treat as percentage (0.5 = 50%)
    - If integer >= 1: treat as step count, convert using steps param
    """
    val = float(when_str)

    if "." in when_str:
        # Percentage
        return max(0.0, min(1.0, val))
    else:
        # Step count
        if steps is not None and steps > 0:
            return max(0.0, min(1.0, val / steps))
        else:
            # No steps provided, treat large values as percentage anyway
            if val >= 1.0:
                return 1.0  # Can't convert without steps
            return val


def _extract_text(item) -> str:
    """Recursively extract text from WeightedPart or nested structures."""
    if isinstance(item, WeightedPart):
        if item.is_alternating and item.alternating_options:
            # For alternating, return options joined (will be handled per-step elsewhere)
            return "|".join(item.alternating_options)
        return item.text
    elif isinstance(item, list):
        return "".join(_extract_text(x) for x in item)
    elif HAS_LARK and isinstance(item, lark.Token):
        return str(item)
    elif item is None:
        return ""
    else:
        return str(item)


def _extract_parts_from_tree(tree, steps: Optional[int]) -> List[WeightedPart]:
    """Extract WeightedParts from parsed Lark tree."""
    parts = []

    class PartExtractor(lark.Transformer):
        def plain(self, args):
            text = args[0].value if args else ""
            return WeightedPart(text=text, weight=1.0)

        def emphasized(self, args):
            if len(args) == 1:
                # Just (text) or [text]
                inner = args[0]
                if isinstance(inner, WeightedPart):
                    inner.weight *= 1.1
                    return inner
                return WeightedPart(text=_extract_text(inner), weight=1.1)
            elif len(args) == 2:
                # (text:weight)
                inner, weight_part = args
                try:
                    w = float(_extract_text(weight_part))
                except:
                    w = 1.0
                if isinstance(inner, WeightedPart):
                    inner.weight *= w
                    return inner
                return WeightedPart(text=_extract_text(inner), weight=w)
            return args[0] if args else WeightedPart(text="")

        def scheduled(self, args):
            # [before:after:when] or [:after:when] or [before::when]
            # args structure varies
            when_str = _extract_text(args[-2]) if len(args) >= 2 else "0.5"
            when_pct = _convert_when_to_percent(when_str, steps)

            if len(args) >= 3:
                before = args[0] or ""
                after = args[1] or ""
            else:
                before = ""
                after = args[0] or ""

            # Create two parts: before and after
            result = []
            if before:
                result.append(
                    WeightedPart(
                        text=_extract_text(before),
                        start_percent=0.0,
                        end_percent=when_pct,
                    )
                )
            if after:
                result.append(
                    WeightedPart(
                        text=_extract_text(after),
                        start_percent=when_pct,
                        end_percent=1.0,
                    )
                )
            return result

        def alternate(self, args):
            options = [_extract_text(a) if a else "" for a in args]
            return WeightedPart(
                text="", is_alternating=True, alternating_options=options
            )

        def prompt(self, args):
            return args

        def start(self, args):
            return args

    try:
        result = PartExtractor().transform(tree)

        # Flatten result
        def flatten(items):
            for item in items:
                if isinstance(item, list):
                    yield from flatten(item)
                elif isinstance(item, WeightedPart):
                    yield item
                elif item is not None:
                    yield WeightedPart(text=str(item))

        parts = list(flatten(result if isinstance(result, list) else [result]))
    except Exception:
        parts = [WeightedPart(text=tree)]

    return parts if parts else [WeightedPart(text="")]


# ============ SIMPLE FALLBACK PARSER (no Lark) ============

# Regex patterns for simple parser
re_schedule_switch = re.compile(r"^\[([^:\[\]]*):([^:\[\]]*):([0-9.]+)\]$")
re_schedule_remove = re.compile(r"^\[([^:\[\]]*)::([0-9.]+)\]$")
re_schedule_add = re.compile(r"^\[([^:\[\]]*):([0-9.]+)\]$")
re_alternation = re.compile(r"^\[([^|\]]+(?:\|[^|\]]+)+)\]$")

re_attention = re.compile(
    r"""
\\\\|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\[[^\]|]*\|[^\]|]*\]|
\[[^\]:]*:[^\]]*\]|
\(|
\[|
:\s*([+-]?[.\d]+)\s*\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)


def _parse_chunk_simple(chunk: str, steps: Optional[int]) -> List[WeightedPart]:
    """Simple regex-based parsing (fallback when Lark not available)."""
    parsed_weights = _parse_weight_simple(chunk)
    parts = []

    for text, weight in parsed_weights:
        # Check for scheduling patterns
        m_switch = re_schedule_switch.search(text)
        if m_switch:
            p_from, p_to, p_when = m_switch.groups()
            val = _convert_when_to_percent(p_when, steps)
            parts.append(
                WeightedPart(
                    text=p_from, weight=weight, start_percent=0.0, end_percent=val
                )
            )
            parts.append(
                WeightedPart(
                    text=p_to, weight=weight, start_percent=val, end_percent=1.0
                )
            )
            continue

        m_remove = re_schedule_remove.search(text)
        if m_remove:
            p_from, p_when = m_remove.groups()
            val = _convert_when_to_percent(p_when, steps)
            parts.append(
                WeightedPart(
                    text=p_from, weight=weight, start_percent=0.0, end_percent=val
                )
            )
            continue

        m_add = re_schedule_add.search(text)
        if m_add:
            p_to, p_when = m_add.groups()
            val = _convert_when_to_percent(p_when, steps)
            parts.append(
                WeightedPart(
                    text=p_to, weight=weight, start_percent=val, end_percent=1.0
                )
            )
            continue

        m_alt = re_alternation.search(text)
        if m_alt:
            options = m_alt.group(1).split("|")
            parts.append(
                WeightedPart(
                    text=text,
                    weight=weight,
                    is_alternating=True,
                    alternating_options=options,
                )
            )
            continue

        parts.append(WeightedPart(text=text, weight=weight))

    return parts if parts else [WeightedPart(text="")]


def _parse_weight_simple(text: str) -> List[tuple]:
    """Parse emphasis weights using regex."""
    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        token = m.group(0)
        weight = m.group(1)

        if token.startswith("\\"):
            res.append([token[1:], 1.0])
        elif (token.startswith("[") and "|" in token and token.endswith("]")) or (
            token.startswith("[") and ":" in token and token.endswith("]")
        ):
            res.append([token, 1.0])
        elif token == "(":
            round_brackets.append(len(res))
        elif token == "[":
            square_brackets.append(len(res))
        elif weight is not None and round_brackets:
            multiply_range(round_brackets.pop(), float(weight))
        elif token == ")" and round_brackets:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif token == "]" and square_brackets:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            res.append([token, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)
    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if not res:
        res = [["", 1.0]]

    # Merge consecutive identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return [(t, w) for t, w in res]
