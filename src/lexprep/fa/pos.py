"""Persian POS tagging using Stanza.

Uses Stanford NLP's Stanza library for Universal Dependencies POS tagging.

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

_nlp_cache = None


@dataclass
class POSTagResult:
    """Result of POS tagging a Persian word."""
    word: str
    normalized: str
    length: int
    pos_tag: Optional[str]
    xpos: Optional[str] = None
    lemma: Optional[str] = None
    error: Optional[str] = None


def _persian_len(s: str) -> int:
    """Get length of Persian text, ignoring ZWNJ and spaces."""
    return len(s.replace("\u200c", "").replace(" ", ""))


def _get_pipeline():
    """Get or create the Stanza pipeline (cached)."""
    global _nlp_cache
    if _nlp_cache is None:
        try:
            import stanza
        except ImportError as e:
            raise ImportError(
                "Persian POS requires Stanza. Install with: pip install 'lexprep[fa]'"
            ) from e

        try:
            _nlp_cache = stanza.Pipeline(
                lang="fa",
                processors="tokenize,pos,lemma",
                tokenize_pretokenized=True,
                verbose=False
            )
        except Exception:
            stanza.download("fa", verbose=False)
            _nlp_cache = stanza.Pipeline(
                lang="fa",
                processors="tokenize,pos,lemma",
                tokenize_pretokenized=True,
                verbose=False
            )
    return _nlp_cache


def tag_words(
    words: List[str],
    *,
    min_length: int = 1,
    max_length: int = 999,
) -> List[POSTagResult]:
    """Tag Persian words using Stanza."""
    nlp = _get_pipeline()

    results: List[POSTagResult] = []
    for w in words:
        raw = str(w).strip()
        if not raw:
            continue

        try:
            doc = nlp([[raw]])

            if doc.sentences and doc.sentences[0].words:
                word_obj = doc.sentences[0].words[0]
                normalized = word_obj.text
                word_len = _persian_len(normalized)

                if word_len < min_length or word_len > max_length:
                    results.append(POSTagResult(
                        word=raw,
                        normalized=normalized,
                        length=word_len,
                        pos_tag=None,
                        error="filtered"
                    ))
                    continue

                results.append(POSTagResult(
                    word=raw,
                    normalized=normalized,
                    length=word_len,
                    pos_tag=word_obj.upos,
                    xpos=word_obj.xpos,
                    lemma=word_obj.lemma
                ))
            else:
                results.append(POSTagResult(
                    word=raw,
                    normalized=raw,
                    length=_persian_len(raw),
                    pos_tag=None,
                    error="no_result"
                ))

        except Exception as e:
            results.append(POSTagResult(
                word=raw,
                normalized=raw,
                length=_persian_len(raw),
                pos_tag=None,
                error=str(e)
            ))

    return results


def tag_words_batch(
    words: List[str],
    *,
    batch_size: int = 100,
    min_length: int = 1,
    max_length: int = 999,
) -> List[POSTagResult]:
    """Tag Persian words using Stanza in batches."""
    nlp = _get_pipeline()

    non_empty = []
    indices = []
    for i, w in enumerate(words):
        raw = str(w).strip()
        if raw:
            non_empty.append(raw)
            indices.append(i)

    results = [None] * len(words)

    for batch_start in range(0, len(non_empty), batch_size):
        batch = non_empty[batch_start:batch_start + batch_size]
        batch_indices = indices[batch_start:batch_start + batch_size]

        try:
            doc = nlp([[w] for w in batch])

            for sent, orig_word, orig_idx in zip(doc.sentences, batch, batch_indices):
                if sent.words:
                    word_obj = sent.words[0]
                    normalized = word_obj.text
                    word_len = _persian_len(normalized)

                    if word_len < min_length or word_len > max_length:
                        results[orig_idx] = POSTagResult(
                            word=orig_word,
                            normalized=normalized,
                            length=word_len,
                            pos_tag=None,
                            error="filtered"
                        )
                    else:
                        results[orig_idx] = POSTagResult(
                            word=orig_word,
                            normalized=normalized,
                            length=word_len,
                            pos_tag=word_obj.upos,
                            xpos=word_obj.xpos,
                            lemma=word_obj.lemma
                        )
                else:
                    results[orig_idx] = POSTagResult(
                        word=orig_word,
                        normalized=orig_word,
                        length=_persian_len(orig_word),
                        pos_tag=None,
                        error="no_result"
                    )

        except Exception as e:
            for orig_word, orig_idx in zip(batch, batch_indices):
                results[orig_idx] = POSTagResult(
                    word=orig_word,
                    normalized=orig_word,
                    length=_persian_len(orig_word),
                    pos_tag=None,
                    error=str(e)
                )

    return [r for r in results if r is not None]
