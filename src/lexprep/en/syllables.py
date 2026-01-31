from __future__ import annotations

from typing import List


def count_syllables(word: str) -> int:
    """Count syllables in an English word using pyphen."""
    try:
        import pyphen
    except ImportError as e:
        raise ImportError(
            "English syllables requires: pip install 'lexprep[en]'"
        ) from e

    w = str(word).strip().lower()
    if not w:
        return 0

    dic = pyphen.Pyphen(lang='en_US')
    hyphenated = dic.inserted(w)
    return hyphenated.count('-') + 1 if hyphenated else 1


def count_syllables_batch(words: List[str]) -> List[int]:
    """Count syllables for multiple words."""
    return [count_syllables(w) for w in words]
