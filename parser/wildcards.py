"""
Wildcard expansion helpers for A1111-style prompt text.

Supported syntax:
- ``__wildcard__`` for files like ``data/wildcards/wildcard.txt``
- Nested directories via dot notation, e.g. ``__characters.anime__``
- Nested wildcard expansion inside selected lines
- Dynamic prompt choices like ``{a|b}``, ``{1-2$$a|b|c}``, and ``{20%a|b}``
"""

from __future__ import annotations

import logging
import random
import re
from pathlib import Path
from typing import Callable, Iterable, List, Optional

logger = logging.getLogger("A1111PromptNode")

DEFAULT_WILDCARDS_DIR = Path(__file__).resolve().parent.parent / "data" / "wildcards"
WILDCARD_PATTERN = re.compile(r"__([^_]+(?:_[^_]+)*)__")
DYNAMIC_PROMPT_PATTERN = re.compile(r"\{([^{}]*)\}")
MAX_WILDCARD_EXPANSION_DEPTH = 32


def _get_wildcards_dir(wildcards_dir: Optional[str] = None) -> Path:
    return Path(wildcards_dir) if wildcards_dir else DEFAULT_WILDCARDS_DIR


def _iter_wildcard_files(wildcards_dir: Path) -> Iterable[Path]:
    if not wildcards_dir.exists():
        return []
    return wildcards_dir.rglob("*.txt")


def _display_name_for_file(path: Path, wildcards_dir: Path) -> str:
    relative = path.relative_to(wildcards_dir).with_suffix("")
    return ".".join(relative.parts)


def _normalize_wildcard_name(name: str) -> str:
    return name.strip().replace("\\", ".").replace("/", ".").strip(".").lower()


def _resolve_wildcard_file(
    wildcard_name: str, wildcards_dir: Optional[str] = None
) -> Optional[Path]:
    base_dir = _get_wildcards_dir(wildcards_dir)
    if not base_dir.exists():
        return None

    normalized_name = _normalize_wildcard_name(wildcard_name)
    if not normalized_name:
        return None

    direct_path = base_dir.joinpath(*normalized_name.split(".")).with_suffix(".txt")
    if direct_path.is_file():
        return direct_path

    for file_path in _iter_wildcard_files(base_dir):
        if _normalize_wildcard_name(_display_name_for_file(file_path, base_dir)) == normalized_name:
            return file_path

    leaf_matches = [
        file_path
        for file_path in _iter_wildcard_files(base_dir)
        if file_path.stem.lower() == normalized_name
    ]
    if len(leaf_matches) == 1:
        return leaf_matches[0]
    if len(leaf_matches) > 1:
        logger.warning(
            "[A1111 Prompt] Ambiguous leaf wildcard name '%s'; use the full path to disambiguate.",
            wildcard_name,
        )

    return None


def _read_wildcard_lines(path: Path) -> List[str]:
    options: List[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line and not line.startswith("#"):
                options.append(line)
    return options


def _get_dynamic_variant_weight(variant: str) -> int:
    split_variant = variant.split("%", 1)
    if len(split_variant) == 2:
        try:
            return int(split_variant[0])
        except ValueError:
            return 0
    return 0


def _strip_dynamic_variant_weight(variant: str) -> str:
    split_variant = variant.split("%", 1)
    if len(split_variant) == 2:
        return split_variant[1]
    return variant


def _parse_dynamic_range(range_str: Optional[str], num_variants: int) -> tuple[int, int]:
    if range_str is None:
        return 1, 1

    parts = range_str.split("-")
    if len(parts) == 1:
        value = min(int(parts[0]), num_variants)
        return value, value
    if len(parts) == 2:
        low = int(parts[0]) if parts[0] else 0
        high = min(int(parts[1]), num_variants) if parts[1] else num_variants
        return min(low, high), max(low, high)
    raise ValueError(f"Unexpected dynamic prompt range: {range_str}")


def _expand_dynamic_prompt_match(
    match: re.Match[str], chooser: random.Random | random.Random
) -> str:
    combinations_str = match.group(1)
    variants = [segment.strip() for segment in combinations_str.split("|")]
    if not variants:
        return ""

    weights = [_get_dynamic_variant_weight(variant) for variant in variants]
    variants = [_strip_dynamic_variant_weight(variant) for variant in variants]

    splits = variants[0].split("$$", 1)
    quantity_spec: Optional[str] = None
    if len(splits) == 2:
        quantity_spec = splits[0].strip()
        variants[0] = splits[1].strip()

    try:
        low_range, high_range = _parse_dynamic_range(quantity_spec, len(variants))
    except ValueError:
        return match.group(0)

    if high_range <= 0:
        return ""

    quantity = chooser.randint(low_range, high_range)
    if quantity <= 0:
        return ""

    total_weight = sum(weights)
    zero_weight_count = weights.count(0)
    if zero_weight_count > 0 and total_weight < 100:
        remaining_weight = max(0, 100 - total_weight)
        fill_weight = remaining_weight / zero_weight_count if zero_weight_count else 0
        weights = [fill_weight if weight == 0 else weight for weight in weights]
    elif all(weight == 0 for weight in weights):
        weights = [1] * len(weights)

    available_variants = list(variants)
    available_weights = list(weights)
    picked: List[str] = []

    for _ in range(min(quantity, len(available_variants))):
        choice = chooser.choices(available_variants, weights=available_weights, k=1)[0]
        picked.append(choice)
        index = available_variants.index(choice)
        available_variants.pop(index)
        available_weights.pop(index)

    return ", ".join(picked)


def _expand_dynamic_prompts(
    text: str, chooser: random.Random | random.Random
) -> tuple[str, bool]:
    changed = False

    def replace_match(match: re.Match[str]) -> str:
        nonlocal changed
        replacement = _expand_dynamic_prompt_match(match, chooser)
        if replacement != match.group(0):
            changed = True
        return replacement

    return DYNAMIC_PROMPT_PATTERN.sub(replace_match, text), changed


def _select_highest_scoring_text(
    options: List[str], scorer: Callable[[str], int]
) -> str:
    if not options:
        return ""
    return max(options, key=lambda option: (scorer(option), len(option), option))


def _expand_dynamic_prompt_match_max(
    match: re.Match[str],
    scorer: Callable[[str], int],
    expander: Callable[[str], str],
) -> str:
    combinations_str = match.group(1)
    variants = [segment.strip() for segment in combinations_str.split("|")]
    if not variants:
        return ""

    variants = [_strip_dynamic_variant_weight(variant) for variant in variants]

    splits = variants[0].split("$$", 1)
    quantity_spec: Optional[str] = None
    if len(splits) == 2:
        quantity_spec = splits[0].strip()
        variants[0] = splits[1].strip()

    try:
        _, high_range = _parse_dynamic_range(quantity_spec, len(variants))
    except ValueError:
        return match.group(0)

    if high_range <= 0:
        return ""

    expanded_variants = [expander(variant) for variant in variants]
    ranked_variants = sorted(
        expanded_variants,
        key=lambda variant: (scorer(variant), len(variant), variant),
        reverse=True,
    )
    return ", ".join(ranked_variants[: min(high_range, len(ranked_variants))])


def _expand_wildcards_for_token_count(
    text: str,
    wildcards_dir: Optional[str],
    scorer: Callable[[str], int],
    depth_remaining: int,
) -> str:
    if not text or depth_remaining <= 0:
        return text

    current = text

    for _ in range(depth_remaining):
        changed = False

        def replace_wildcard(match: re.Match[str]) -> str:
            nonlocal changed
            options = get_wildcard_options(match.group(1), wildcards_dir=wildcards_dir)
            if not options:
                return match.group(0)

            changed = True
            expanded_options = [
                _expand_wildcards_for_token_count(
                    option, wildcards_dir, scorer, depth_remaining - 1
                )
                for option in options
            ]
            return _select_highest_scoring_text(expanded_options, scorer)

        expanded = WILDCARD_PATTERN.sub(replace_wildcard, current)

        def replace_dynamic(match: re.Match[str]) -> str:
            nonlocal changed

            def expand_variant(variant: str) -> str:
                return _expand_wildcards_for_token_count(
                    variant, wildcards_dir, scorer, depth_remaining - 1
                )

            replacement = _expand_dynamic_prompt_match_max(
                match,
                scorer=scorer,
                expander=expand_variant,
            )
            if replacement != match.group(0):
                changed = True
            return replacement

        expanded = DYNAMIC_PROMPT_PATTERN.sub(replace_dynamic, expanded)
        expanded = _normalize_prompt_spacing(expanded)
        if not changed or expanded == current:
            return expanded
        current = expanded

    return current


def _normalize_prompt_spacing(text: str) -> str:
    text = re.sub(r"[ \t]*,[ \t]*", ", ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def expand_wildcards(
    text: str,
    wildcards_dir: Optional[str] = None,
    rng: Optional[random.Random] = None,
    max_depth: int = MAX_WILDCARD_EXPANSION_DEPTH,
) -> str:
    """
    Expand ``__wildcard__`` syntax in prompt text.

    Missing wildcard files are left untouched so the user can spot the problem
    in the prompt preview instead of silently losing text.
    """
    if not text:
        return text

    chooser = rng or random
    current = text

    for _ in range(max_depth):
        changed = False

        def replace_match(match: re.Match[str]) -> str:
            nonlocal changed
            options = get_wildcard_options(match.group(1), wildcards_dir=wildcards_dir)
            if not options:
                return match.group(0)
            changed = True
            return chooser.choice(options)

        expanded = WILDCARD_PATTERN.sub(replace_match, current)
        expanded, dynamic_changed = _expand_dynamic_prompts(expanded, chooser)
        expanded = _normalize_prompt_spacing(expanded)
        changed = changed or dynamic_changed
        if not changed or expanded == current:
            return expanded
        current = expanded

    if WILDCARD_PATTERN.search(current):
        logger.warning(
            "[A1111 Prompt] Wildcard expansion depth limit reached; possible recursive loop in prompt: %s",
            current[:200],
        )
    return current


def expand_wildcards_for_token_count(
    text: str,
    wildcards_dir: Optional[str] = None,
    scorer: Optional[Callable[[str], int]] = None,
    max_depth: int = MAX_WILDCARD_EXPANSION_DEPTH,
) -> str:
    """
    Deterministically expand wildcards and dynamic prompts for worst-case token counting.

    Runtime wildcard expansion remains random. This helper is only for UI estimates
    and chooses the highest-scoring option at each wildcard/dynamic prompt branch.
    """
    score_text = scorer or (lambda value: len(value))
    return _expand_wildcards_for_token_count(
        text,
        wildcards_dir=wildcards_dir,
        scorer=score_text,
        depth_remaining=max_depth,
    )


def get_wildcard_options(
    wildcard_name: str, wildcards_dir: Optional[str] = None
) -> List[str]:
    """
    Get all options for a given wildcard file.

    Returns an empty list when the wildcard file does not exist.
    """
    resolved_file = _resolve_wildcard_file(wildcard_name, wildcards_dir=wildcards_dir)
    if resolved_file is None:
        return []
    return _read_wildcard_lines(resolved_file)


def list_available_wildcards(wildcards_dir: Optional[str] = None) -> List[str]:
    """
    List wildcard names available under the configured wildcard directory.
    """
    base_dir = _get_wildcards_dir(wildcards_dir)
    if not base_dir.exists():
        return []

    wildcard_names = [
        _display_name_for_file(file_path, base_dir)
        for file_path in _iter_wildcard_files(base_dir)
    ]
    return sorted(wildcard_names, key=lambda name: name.lower())
