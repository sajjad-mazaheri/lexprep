from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .pos_map import map_pos_to_english


@dataclass
class UniDicResult:
    text: str
    pos1: str
    pos2: str
    pos3: str
    pos4: str
    lemma: str
    pos_en: str
    error: str | None = None


def tag_with_unidic(texts: List[str]) -> List[UniDicResult]:
    """Tag Japanese texts using UniDic (fugashi)."""
    try:
        from fugashi import Tagger
    except Exception as e:  # pragma: no cover
        raise ImportError("UniDic tagging requires: pip install 'lexprep[ja]'") from e

    tagger = Tagger()
    out: List[UniDicResult] = []

    for t in texts:
        txt = str(t).strip()
        if not txt:
            continue
        try:
            tokens = tagger(txt)
            if not tokens:
                out.append(
                    UniDicResult(
                        text=txt,
                        pos1="UNKNOWN",
                        pos2="",
                        pos3="",
                        pos4="",
                        lemma="",
                        pos_en="Unknown",
                        error=None,
                    )
                )
                continue
            token = tokens[0]
            f = token.feature
            pos1 = getattr(f, "pos1", "")
            out.append(
                UniDicResult(
                    text=txt,
                    pos1=pos1,
                    pos2=getattr(f, "pos2", ""),
                    pos3=getattr(f, "pos3", ""),
                    pos4=getattr(f, "pos4", ""),
                    lemma=getattr(f, "lemma", ""),
                    pos_en=map_pos_to_english(pos1),
                    error=None,
                )
            )
        except Exception as e:
            out.append(
                UniDicResult(
                    text=txt,
                    pos1="ERROR",
                    pos2="",
                    pos3="",
                    pos4="",
                    lemma="",
                    pos_en="Unknown",
                    error=str(e),
                )
            )

    return out
