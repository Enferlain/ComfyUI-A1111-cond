"""
A1111 Prompt Scheduler

Handles step-based scheduling and alternation logic.
Generates the correct prompt text for each step based on A1111 scheduling syntax.
"""

import re
from typing import List, Tuple
import lark
from .grammar import get_parser


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
        parser = get_parser()
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

        def alternate_scheduled(self, tree):
            # Scheduled alternation: [a|b:when] or [a|b::when]
            # Format: options..., maybe ":", maybe ":", NUMBER
            # Find the NUMBER token (should be near the end)
            when_token = None
            has_double_colon = False
            for i, child in enumerate(tree.children):
                if isinstance(child, lark.Token):
                    if child.type == "NUMBER":
                        when_token = child
                    elif str(child) == ":":
                        # Check if there's another colon before the number
                        if when_token is None:
                            # This : comes before NUMBER, check if double
                            for j in range(i + 1, len(tree.children)):
                                if (
                                    isinstance(tree.children[j], lark.Token)
                                    and str(tree.children[j]) == ":"
                                ):
                                    has_double_colon = True
                                    break

            if when_token is not None:
                s_str = str(when_token)
                v = float(s_str)
                if "." in s_str:
                    step_num = int(v * steps)
                else:
                    step_num = int(v)
                step_num = max(1, min(steps, step_num))
                # Store the computed step and whether it's double-colon
                tree.children.append(("_schedule_info", step_num, has_double_colon))
                result.add(step_num)

            # Still need all steps for alternation
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

        def alternate_scheduled(self, args):
            # Scheduled alternation: [a|b:when] or [a|b::when]
            # Extract options and schedule info
            options = []
            schedule_step = None
            is_remove = False  # True for ::when (remove), False for :when (add)

            for arg in args:
                if (
                    isinstance(arg, tuple)
                    and len(arg) == 3
                    and arg[0] == "_schedule_info"
                ):
                    # This is our schedule info from collect_steps
                    schedule_step, is_remove = arg[1], arg[2]
                elif isinstance(arg, lark.Token):
                    # Skip colon and number tokens
                    if arg.type == "NUMBER" or str(arg) == ":":
                        continue
                    options.append(str(arg) if arg else "")
                elif arg is not None:
                    options.append("" if not arg else arg)

            # Fallback: if no schedule info, look for NUMBER token
            if schedule_step is None:
                for arg in args:
                    if isinstance(arg, lark.Token) and arg.type == "NUMBER":
                        s_str = str(arg)
                        v = float(s_str)
                        if "." in s_str:
                            schedule_step = int(v * total_steps)
                        else:
                            schedule_step = int(v)
                        schedule_step = max(1, min(total_steps, schedule_step))
                        # Check for double colon by counting colons
                        colon_count = sum(
                            1
                            for a in args
                            if isinstance(a, lark.Token) and str(a) == ":"
                        )
                        is_remove = colon_count >= 2
                        break

            if schedule_step is None:
                # No schedule found, fall back to normal alternation
                idx = (step - 1) % len(options) if options else 0
                yield options[idx] if options else ""
                return

            # Apply scheduling logic
            # [a|b:when] = nothing before when, alternate after (add at step when)
            # [a|b::when] = alternate before when, nothing after (remove at step when)
            if is_remove:
                # ::when - alternate before, nothing after
                if step <= schedule_step:
                    # Alternate
                    if options:
                        idx = (step - 1) % len(options)
                        yield options[idx]
                    else:
                        yield ""
                else:
                    # After removal, yield nothing
                    yield ""
            else:
                # :when - nothing before, alternate after
                if step <= schedule_step:
                    # Before the schedule point, yield nothing
                    yield ""
                else:
                    # After the schedule point, alternate
                    if options:
                        idx = (step - 1) % len(options)
                        yield options[idx]
                    else:
                        yield ""

        def emphasized(self, args):
            # Grammar uses !emphasized so literal parens/brackets are already in args
            # We just yield them as-is without adding extra parens
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
