from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple


@dataclass(frozen=True)
class BehaviorMappingRule:
    behavior: str
    patterns: Tuple[str, ...]


_DEFAULT_RULES: List[BehaviorMappingRule] = [
    # Phone-related
    BehaviorMappingRule(
        behavior="phone_use",
        patterns=(
            r"texting",
            r"talking on cell phone",
            r"using (a )?phone",
            r"smartphone",
            r"cellphone",
        ),
    ),
    # Writing / note taking
    BehaviorMappingRule(
        behavior="writing",
        patterns=(
            r"writing",
            r"taking notes",
            r"drawing",
            r"sketching",
        ),
    ),
    # Reading
    BehaviorMappingRule(
        behavior="reading",
        patterns=(
            r"reading",
            r"reading book",
            r"reading newspaper",
        ),
    ),
    # Computer / laptop
    BehaviorMappingRule(
        behavior="computer_use",
        patterns=(
            r"typing",
            r"using computer",
            r"working on computer",
            r"laptop",
        ),
    ),
    # Talking (approx.)
    BehaviorMappingRule(
        behavior="talking",
        patterns=(
            r"talking",
            r"speaking",
            r"discussing",
            r"testifying",
            r"answering questions",
            r"giving (a )?speech",
            r"public speaking",
            r"presenting",
            r"lecturing",
        ),
    ),
]


def default_behavior_labels() -> List[str]:
    """Return stable list of classroom behavior labels."""

    # Keep deterministic order for output.
    seen = set()
    labels: List[str] = []
    for r in _DEFAULT_RULES:
        if r.behavior not in seen:
            labels.append(r.behavior)
            seen.add(r.behavior)
    return labels


def build_regex_mapping(
    categories: Iterable[str],
    rules: Iterable[BehaviorMappingRule] = _DEFAULT_RULES,
) -> Dict[str, List[int]]:
    """Build mapping behavior -> list of category indices.

    This maps model categories (e.g., Kinetics-400 class names) to our classroom
    behavior labels using regex rules.
    """

    cats = list(categories)
    out: Dict[str, List[int]] = {}

    compiled: List[Tuple[str, List[re.Pattern[str]]]] = []
    for r in rules:
        compiled.append((r.behavior, [re.compile(p, re.IGNORECASE) for p in r.patterns]))

    for idx, name in enumerate(cats):
        for behavior, pats in compiled:
            if any(p.search(name) for p in pats):
                out.setdefault(behavior, []).append(idx)

    return out


def behaviors_from_category_scores(
    category_scores: Mapping[str, float],
    mapping: Mapping[str, List[str]],
) -> Dict[str, float]:
    """Aggregate category scores into behavior scores.

    Args:
        category_scores: mapping category_name -> probability/score.
        mapping: behavior -> list of category names.

    Returns:
        behavior -> score (max over mapped categories).
    """

    out: Dict[str, float] = {}
    for behavior, cats in mapping.items():
        best = 0.0
        for c in cats:
            s = float(category_scores.get(c, 0.0) or 0.0)
            if s > best:
                best = s
        out[behavior] = best
    return out
