from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class G2PResult:
    word: str
    pronunciation: str
    error: str | None = None


def transcribe_words(
    words: List[str],
    *,
    use_large: bool = True,
    tidy: bool = True,
) -> List[G2PResult]:
    """Convert Persian words to phonemes."""
    try:
        from PersianG2p import Persian_g2p_converter
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Persian G2P requires extra deps. Install with: pip install 'lexprep[fa]'"
        ) from e

    converter = Persian_g2p_converter(use_large=use_large)
    results: List[G2PResult] = []

    for w in words:
        word = str(w).strip()
        if not word:
            continue
        try:
            pron = converter.transliterate(word, tidy=tidy)
            results.append(G2PResult(word=word, pronunciation=pron, error=None))
        except Exception as e:
            results.append(G2PResult(word=word, pronunciation="", error=str(e)))

    return results
