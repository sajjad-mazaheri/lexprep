"English G2P using g2p_en library."
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class G2PResult:
    word: str
    pronunciation: str
    error: str | None = None


def transcribe_words(words: List[str]) -> List[G2PResult]:
    """Convert English words to phonemes (ARPAbet)."""
    try:
        from g2p_en import G2p
    except ImportError as e:
        raise ImportError(
            "English G2P requires: pip install 'lexprep[en]'"
        ) from e

    g2p = G2p()
    results: List[G2PResult] = []

    for w in words:
        word = str(w).strip()
        if not word:
            continue
        try:
            phonemes = g2p(word)
            pron = " ".join(phonemes)
            results.append(G2PResult(word=word, pronunciation=pron, error=None))
        except Exception as e:
            results.append(G2PResult(word=word, pronunciation="", error=str(e)))

    return results
