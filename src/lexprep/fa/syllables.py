from __future__ import annotations

from typing import List, Tuple

DEFAULT_VOWEL_PATTERNS = [
    # Longer patterns matched first for accuracy
    "ā", "ī", "ū", "ey", "ia", "ea", "iā", "eī", "oi", "ya",
    "a", "e", "o", "i", "u",
]

# Persian orthographic vowels (simplified)
PERSIAN_VOWELS = "اآایو"
# Common compound vowel-like sequences encountered in Persian wordlists
DEFAULT_COMPOUND_VOWELS = ["آ", "او", "ای", "آی", "ایَ", "ایِ", "اِی", "آو"]


def count_syllables_from_pronunciation(
    pronunciation: str,
    *,
    vowel_patterns: List[str] | None = None,
) -> int:
    """Count syllables in a Persian pronunciation string."""
    if pronunciation is None:
        return 0
    compact = str(pronunciation).replace(" ", "").strip().lower()
    if not compact:
        return 0

    patterns = vowel_patterns if vowel_patterns else DEFAULT_VOWEL_PATTERNS
    # Ensure longest-first
    patterns = sorted(patterns, key=len, reverse=True)

    count = 0
    i = 0
    while i < len(compact):
        matched = False
        for v in patterns:
            if compact.startswith(v, i):
                count += 1
                i += len(v)
                matched = True
                break
        if not matched:
            i += 1
    return count


def syllabify_orthographic(
    word: str,
    *,
    vowels: str = PERSIAN_VOWELS,
    compound_vowels: List[str] | None = None,
    delimiter: str = "-",
) -> Tuple[str, int]:

    w = str(word).strip()
    if not w:
        return "", 0

    compounds = compound_vowels if compound_vowels else DEFAULT_COMPOUND_VOWELS
    compounds_sorted = sorted(compounds, key=len, reverse=True)

    syllables: List[str] = []
    current = ""
    i = 0

    while i < len(w):
        compound_found = False
        for comp in compounds_sorted:
            if w.startswith(comp, i):
                current += comp
                i += len(comp)
                compound_found = True
                break

        if compound_found:
            # After a compound vowel, attach following consonants until next vowel.
            while i < len(w) and w[i] not in vowels:
                current += w[i]
                i += 1
            syllables.append(current)
            current = ""
            continue

        current += w[i]
        if w[i] in vowels:
            i += 1
            while i < len(w) and w[i] not in vowels:
                current += w[i]
                i += 1
            syllables.append(current)
            current = ""
        else:
            i += 1

    if current:
        if syllables:
            syllables[-1] += current
        else:
            syllables.append(current)

    return delimiter.join(syllables), len(syllables)
