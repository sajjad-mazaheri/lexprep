from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class JaPOSResult:
    text: str
    upos: str
    error: str | None = None


def tag_pretokenized_with_stanza(
    tokens: List[str],
    *,
    download_if_missing: bool = False,
) -> List[JaPOSResult]:

    try:
        import stanza
    except Exception as e:  # pragma: no cover
        raise ImportError("Japanese Stanza tagging requires: pip install 'lexprep[ja]'") from e

    if download_if_missing:
        stanza.download("ja")

    nlp = stanza.Pipeline(lang="ja", processors="tokenize,pos", tokenize_pretokenized=True)

    out: List[JaPOSResult] = []
    for t in tokens:
        txt = str(t).strip()
        if not txt:
            continue
        try:
            doc = nlp([[txt]])
            has_words = doc.sentences and doc.sentences[0].words
            upos = doc.sentences[0].words[0].upos if has_words else "UNKNOWN"
            out.append(JaPOSResult(text=txt, upos=upos, error=None))
        except Exception as e:
            out.append(JaPOSResult(text=txt, upos="ERROR", error=str(e)))
    return out
