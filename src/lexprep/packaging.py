"""ZIP packaging for reproducibility packs (D1)."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .manifest import format_timestamp_filename, sanitize_basename


def make_zip_filename(
    input_basename: str,
    tool: str,
    language: Optional[str],
    timestamp: datetime,
) -> str:
    """Build ZIP filename per spec.

    Pattern: ``{basename}__{tool}__{language}__{YYYYMMDDTHHMMSSZ}.zip``
    """
    safe_base = sanitize_basename(input_basename)
    safe_tool = sanitize_basename(tool)
    ts_str = format_timestamp_filename(timestamp)
    lang = sanitize_basename(language) if language else "all"
    return f"{safe_base}__{safe_tool}__{lang}__{ts_str}.zip"


def make_output_filename(input_basename: str, ext: str, *, is_sampling: bool = False) -> str:
    """Build main output filename inside the ZIP.

    Language tools  → ``{basename}__enriched.{ext}``
    Sampling        → ``{basename}__sample.{ext}``
    """
    safe_base = sanitize_basename(input_basename)
    suffix = "__sample" if is_sampling else "__enriched"
    return f"{safe_base}{suffix}.{ext}"


# -----------------------------------------------------------------------
# DataFrame serialisation
# -----------------------------------------------------------------------


def _df_to_bytes(df: pd.DataFrame, ext: str) -> bytes:
    """Serialise a DataFrame to bytes in the given format."""
    buf = io.BytesIO()
    if ext == "csv":
        df.to_csv(buf, index=False, encoding="utf-8-sig")
    elif ext == "tsv":
        df.to_csv(buf, index=False, sep="\t", encoding="utf-8-sig")
    else:  # xlsx
        df.to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    return buf.getvalue()


# -----------------------------------------------------------------------
# ZIP builder
# -----------------------------------------------------------------------


def build_zip(
    *,
    manifest: Dict[str, Any],
    main_df: pd.DataFrame,
    input_basename: str,
    output_ext: str,
    is_sampling: bool = False,
    excluded_df: Optional[pd.DataFrame] = None,
    audit_bytes: Optional[bytes] = None,
    extra_files: Optional[List[Tuple[str, bytes]]] = None,
) -> Tuple[bytes, str]:
    """Build a ZIP reproducibility pack.

    Files are written in a deterministic, fixed order so that identical
    inputs always produce the same archive structure.

    Returns
    -------
    (zip_bytes, zip_filename)
    """
    tool = manifest.get("tool", "unknown")
    language = manifest.get("language")
    try:
        ts = datetime.fromisoformat(manifest["timestamp_utc"])
    except (ValueError, TypeError, KeyError):
        ts = datetime.utcnow()

    zip_name = make_zip_filename(input_basename, tool, language, ts)
    main_name = make_output_filename(input_basename, output_ext, is_sampling=is_sampling)
    safe_base = sanitize_basename(input_basename)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. run_manifest.json (always first)
        zf.writestr(
            "run_manifest.json",
            json.dumps(manifest, indent=2, ensure_ascii=False),
        )

        # 2. Main output file (skip if empty, e.g. shuffle tools)
        if not main_df.empty:
            zf.writestr(main_name, _df_to_bytes(main_df, output_ext))

        # 3. sampling_audit.xlsx (sampling only — deterministic order)
        if audit_bytes is not None:
            zf.writestr("sampling_audit.xlsx", audit_bytes)

        # 4. excluded file (sampling only)
        if excluded_df is not None:
            excluded_name = f"{safe_base}__excluded.xlsx"
            zf.writestr(excluded_name, _df_to_bytes(excluded_df, "xlsx"))

        # 5. Extra files (e.g. shuffle outputs) — sorted for determinism
        if extra_files:
            for fname, fbytes in sorted(extra_files, key=lambda x: x[0]):
                zf.writestr(fname, fbytes)

    buf.seek(0)
    return buf.getvalue(), zip_name
