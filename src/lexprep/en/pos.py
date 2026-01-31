"""English POS tagging using spaCy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class POSResult:
    word: str
    pos: str
    tag: str
    lemma: str
    error: str | None = None


def tag_words(words: List[str], model: str = "en_core_web_sm") -> List[POSResult]:
    """Tag English words with part-of-speech using spaCy."""
    try:
        import spacy
    except ImportError as e:
        raise ImportError(
            "English POS requires: pip install 'lexprep[en]'"
        ) from e

    try:
        nlp = spacy.load(model)
    except OSError:
        raise RuntimeError(
            f"spaCy model '{model}' not found. Run: python -m spacy download {model}"
        )

    results: List[POSResult] = []
    for w in words:
        word = str(w).strip()
        if not word:
            continue
        try:
            doc = nlp(word)
            if doc:
                token = doc[0]
                results.append(POSResult(
                    word=word,
                    pos=token.pos_,
                    tag=token.tag_,
                    lemma=token.lemma_,
                    error=None
                ))
            else:
                results.append(POSResult(word=word, pos="", tag="", lemma="", error="empty"))
        except Exception as e:
            results.append(POSResult(word=word, pos="", tag="", lemma="", error=str(e)))

    return results
