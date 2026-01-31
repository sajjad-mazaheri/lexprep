from __future__ import annotations

from typing import Dict

DEFAULT_POS_MAP: Dict[str, str] = {
    "名詞": "noun",
    "動詞": "verb",
    "形容詞": "adjective",
    "副詞": "adverb",
    "助詞": "particle",
    "助動詞": "auxiliary verb",
    "連体詞": "prenoun adjectival",
    "感動詞": "interjection",
    "記号": "symbol",
    "接続詞": "conjunction",
    "接頭辞": "prefix",
    "接尾辞": "suffix",
    "フィラー": "filler",
    "その他": "other",
    "補助記号": "supplementary symbol",
    "代名詞": "pronoun",
    "形状詞": "adjectival noun",
}


def normalize_pos_japanese(pos: str) -> str:
    if pos is None:
        return ""
    return str(pos).replace("\u3000", "").replace("　", "").replace("\n", "").strip()


def map_pos_to_english(pos: str, *, mapping: Dict[str, str] | None = None) -> str:
    mp = mapping if mapping is not None else DEFAULT_POS_MAP
    key = normalize_pos_japanese(pos)
    return mp.get(key, "Unknown")
