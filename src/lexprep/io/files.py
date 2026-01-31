from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

SUPPORTED_TABULAR_EXTS = {".csv", ".tsv", ".xlsx", ".xls"}
SUPPORTED_TEXT_EXTS = {".txt"}


@dataclass(frozen=True)
class WordlistSpec:
    word_col: str = "word"
    pronunciation_col: str = "pronunciation"


def _guess_word_column(df: pd.DataFrame) -> str:
    # Common candidates
    candidates = ["word", "Word", "Item", "text", "Text", "token", "Token"]
    for c in candidates:
        if c in df.columns:
            return c
    # Fallback: first non-empty column
    if len(df.columns) == 0:
        raise ValueError("Input table has no columns.")
    return str(df.columns[0])


def read_table(path: str | Path, *, sheet: Optional[str] = None, skiprows: int = 0) -> pd.DataFrame:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig", skiprows=skiprows)
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t", encoding="utf-8-sig", skiprows=skiprows)
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet if sheet else 0, skiprows=skiprows)
    supported = sorted(SUPPORTED_TABULAR_EXTS)
    raise ValueError(f"Unsupported tabular file: {path} (supported: {supported})")


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".csv":
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return
    if ext == ".tsv":
        df.to_csv(path, index=False, sep="\t", encoding="utf-8-sig")
        return
    if ext in {".xlsx", ".xls"}:
        df.to_excel(path, index=False)
        return
    raise ValueError(f"Unsupported output format: {path} (use .csv/.tsv/.xlsx)")


def read_wordlist(
    path: str | Path,
    *,
    word_col: Optional[str] = None,
    sheet: Optional[str] = None,
    skiprows: int = 0,
) -> Tuple[pd.DataFrame, str]:

    path = Path(path)
    ext = path.suffix.lower()

    if ext in SUPPORTED_TEXT_EXTS:
        with path.open("r", encoding="utf-8") as f:
            words = [ln.strip() for ln in f.readlines() if ln.strip()]
        df = pd.DataFrame({"word": words})
        return df, "word"

    df = read_table(path, sheet=sheet, skiprows=skiprows)
    col = word_col if word_col else _guess_word_column(df)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found. Available columns: {list(df.columns)}")
    return df, col


def coerce_str_series(s: pd.Series) -> pd.Series:
    return s.astype(str).fillna("").map(lambda x: x.strip())


def ensure_column(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if name not in df.columns:
        df[name] = ""
    return df
