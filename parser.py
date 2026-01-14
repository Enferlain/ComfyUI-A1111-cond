"""
A1111 Prompt Parser

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
from typing import List, Tuple
import lark


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
plain: /([^\\[\]():|]|\\.)+/
%import common.SIGNED_NUMBER -> NUMBER
"""

# Cached parser instance
_parser = None


def _get_parser():
    """Get or create cached Lark parser."""
    global _parser
    if _parser is None:
        _parser = lark.Lark(GRAMMAR)
    return _parser


def get_prompt_schedule(prompt: str, steps: int) -> List[Tuple[int, str]]:
    """
    A1111-style prompt schedule generation.

    Returns list of (end_at_step, prompt_text) tuples representing
    when each prompt variant should be used.

    This properly handles nested alternation and scheduling.
    """
    # Split by BREAK first, process each chunk, then rejoin
    break_chunks = re.split(r"\s*\bBREAK\b\s*", prompt)

    try:
        parser = _get_parser()
        trees = [parser.parse(chunk) for chunk in break_chunks]
    except lark.exceptions.LarkError:
        # Parse error - return original prompt for all steps
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
    """Collect all steps where the prompt changes.

    A1111 behavior: modifies tree.children[-2] in place to be the computed step number.
    This is critical for at_step to work correctly.
    """
    result = {steps}  # Always include final step

    class CollectSteps(lark.Visitor):
        def scheduled(self, tree):
            # A1111 uses tree.children[-2] directly (the NUMBER token)
            s = tree.children[-2]
            if isinstance(s, lark.Token):
                s_str = str(s)
                v = float(s_str)
                if "." in s_str:
                    step_num = int(v * steps)
                else:
                    step_num = int(v)
            else:
                # Already converted by outer scheduling
                step_num = int(s)

            step_num = max(1, min(steps, step_num))
            # A1111 CRITICAL: modify tree in place so at_step gets integer
            tree.children[-2] = step_num
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
            # A1111 uses strict 5-positional unpacking:
            # before, after, _, when, _ = args
            # where 'when' is already an integer (modified by collect_steps)

            # Pad args to 5 elements for consistent unpacking
            padded = list(args) + [None] * max(0, 5 - len(args))
            before, after, _, when, _ = padded[:5]

            # 'when' should be an integer now (set by collect_steps)
            # If it's still a token, convert it
            if isinstance(when, lark.Token):
                when_str = str(when)
                if "." in when_str:
                    when = int(float(when_str) * total_steps)
                else:
                    when = int(float(when_str))
            elif when is None:
                # No valid when found, yield everything
                for a in args:
                    if a is not None:
                        yield a
                return

            # A1111: yield before if step <= when, else yield after
            # Use 'or ()' pattern like A1111 for empty handling
            if step <= when:
                yield before or ()
            else:
                if after is not None:
                    yield after

        def alternate(self, args):
            # A1111: Convert falsy to "" but KEEP them (affects cycle count)
            # Don't filter out empty - "[A|]" should alternate A, "", A, ""...
            options = ["" if not a else a for a in args]
            if not options:
                yield ""
                return
            idx = (step - 1) % len(options)
            yield options[idx]

        def emphasized(self, args):
            # Keep emphasis syntax in output
            if len(args) == 1:
                # [text] de-emphasis - preserve brackets for tokenizer
                yield "["
                for a in args:
                    yield a
                yield "]"
            elif len(args) >= 2:
                # (text:weight) - preserve weight
                yield "("
                for a in args:
                    if isinstance(a, str):
                        yield a
                    elif isinstance(a, lark.Token):
                        yield str(a)
                    elif hasattr(a, "__iter__"):
                        for item in a:
                            yield item
                    elif a is not None:
                        yield str(a)
                yield ")"
            else:
                for a in args:
                    yield a

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
