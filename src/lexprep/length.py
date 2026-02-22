"""Length feature: adds length_chars column (Unicode codepoint count)."""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from typing import List, Optional

# Documented method for reproducibility
LENGTH_METHOD = "unicode_codepoints"


@dataclass
class LengthDistribution:
    """Summary statistics for token lengths."""

    min: int
    max: int
    mean: float
    median: float

    def to_dict(self) -> dict:
        return asdict(self)


def compute_length_chars(words: List[str]) -> List[int]:
    """Return Unicode codepoint count for each word.

    Uses Python ``len()`` which counts Unicode codepoints (not bytes).
    Empty or whitespace-only tokens receive length 0.
    """
    return [len(w) if w and w.strip() else 0 for w in words]


def length_distribution(lengths: List[int]) -> Optional[LengthDistribution]:
    """Compute summary stats for a list of lengths.

    Returns ``None`` if no valid (>0) lengths exist.
    """
    valid = [ln for ln in lengths if ln > 0]
    if not valid:
        return None
    return LengthDistribution(
        min=min(valid),
        max=max(valid),
        mean=round(statistics.mean(valid), 2),
        median=round(statistics.median(valid), 2),
    )
