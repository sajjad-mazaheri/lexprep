"""Run manifest generation for reproducibility packs (D2)."""

from __future__ import annotations

import importlib.metadata
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import PurePath
from typing import Any, Dict, List, Optional

from lexprep import __version__

# ---------------------------------------------------------------------------
# Tool Registry — single source of truth for tool names + stable columns
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSpec:
    """Describes a tool's manifest metadata."""

    manifest_name: str
    stable_added_columns: tuple[str, ...]
    supports_summary: bool = False
    supports_reproducibility: bool = False


TOOL_REGISTRY: Dict[str, ToolSpec] = {
    # Language tools — Persian
    "fa_g2p": ToolSpec("g2p", ("pronunciation", "g2p_error")),
    "fa_syllables": ToolSpec(
        "syllable_count", ("syllabified", "syllable_count"), supports_summary=True
    ),
    "fa_syllables_phonetic": ToolSpec(
        "syllable_count", ("pronunciation", "syllable_count"), supports_summary=True
    ),
    "fa_pos": ToolSpec("pos_tagging", ("normalized", "pos", "lemma"), supports_summary=True),
    # Language tools — English
    "en_g2p": ToolSpec("g2p", ("pronunciation", "g2p_error")),
    "en_syllables": ToolSpec("syllable_count", ("syllable_count",), supports_summary=True),
    "en_pos": ToolSpec("pos_tagging", ("pos", "tag", "lemma"), supports_summary=True),
    # Language tools — Japanese
    "ja_pos_unidic": ToolSpec(
        "pos_tagging", ("pos", "pos_english", "lemma"), supports_summary=True
    ),
    "ja_pos_stanza": ToolSpec("pos_tagging", ("pos",), supports_summary=True),
    # Universal tools
    "length": ToolSpec("length", ("length_chars",), supports_summary=True),
    # Sampling tools
    "stratified": ToolSpec("stratified_sampling", ("bin_id",), supports_reproducibility=True),
    "shuffle": ToolSpec("row_shuffle", (), supports_reproducibility=True),
}


def registry_key(language: Optional[str], tool: str) -> str:
    """Build a registry key from language + tool."""
    if language:
        key = f"{language}_{tool}"
        if key in TOOL_REGISTRY:
            return key
    # Fall back to tool-only key
    if tool in TOOL_REGISTRY:
        return tool
    return tool


# ---------------------------------------------------------------------------
# Citation — synced with CITATION.cff
# ---------------------------------------------------------------------------

CITATION_INFO = {
    "authors": [
        {
            "family_names": "Mazaherizaveh",
            "given_names": "Sajjad",
            "orcid": "https://orcid.org/0009-0001-0465-0444",
        }
    ],
    "title": "LexPrep",
    "doi": "10.5281/zenodo.18713755",
    "url": "https://github.com/sajjad-mazaheri/lexprep",
    "license": "MIT",
    "version": __version__,
}


# ---------------------------------------------------------------------------
# Timestamp utility
# ---------------------------------------------------------------------------


def utc_now() -> datetime:
    """Return current UTC datetime (single source for consistency)."""
    return datetime.now(timezone.utc)


def format_timestamp_filename(ts: datetime) -> str:
    """Format timestamp for ZIP filenames: ``YYYYMMDDTHHMMSSZ``."""
    return ts.strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------------------
# Basename sanitisation
# ---------------------------------------------------------------------------

_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f\s]+')


def sanitize_basename(name: str) -> str:
    """Remove/replace characters unsafe for filenames and strip path components.

    Ensures that potentially untrusted basenames cannot introduce
    directory traversal or hidden file semantics when used in filenames.
    """
    # Replace clearly unsafe characters (including slashes and whitespace)
    clean = _UNSAFE_CHARS.sub("_", name)
    # Normalise any remaining path structure (e.g. "..", ".") and keep only the name
    clean = PurePath(clean).name
    # Strip leading/trailing underscores and dots to avoid hidden or empty names
    clean = clean.strip("_.")
    return clean or "output"


# ---------------------------------------------------------------------------
# Library version detection
# ---------------------------------------------------------------------------


def get_library_version(package_name: str) -> str:
    """Get installed version of a package via importlib.metadata."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def get_libraries_for_tool(language: Optional[str], tool: str) -> List[Dict[str, str]]:
    """Return list of ``{name, version}`` dicts for the libraries used by a tool."""
    libs: List[Dict[str, str]] = []

    if language == "fa":
        if tool in ("g2p", "syllables_phonetic"):
            libs.append({"name": "PersianG2p", "version": get_library_version("PersianG2p")})
        if tool == "pos":
            libs.append({"name": "stanza", "version": get_library_version("stanza")})
    elif language == "en":
        if tool == "g2p":
            libs.append({"name": "g2p-en", "version": get_library_version("g2p-en")})
        if tool == "syllables":
            libs.append({"name": "pyphen", "version": get_library_version("pyphen")})
        if tool == "pos":
            libs.append({"name": "spacy", "version": get_library_version("spacy")})
    elif language == "ja":
        if tool in ("pos_unidic",):
            libs.append({"name": "fugashi", "version": get_library_version("fugashi")})
            libs.append({"name": "unidic-lite", "version": get_library_version("unidic-lite")})
        if tool in ("pos_stanza",):
            libs.append({"name": "stanza", "version": get_library_version("stanza")})

    if tool == "length":
        pass  # No external libs

    return libs


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------


def build_manifest(
    *,
    tool_key: str,
    language: Optional[str],
    original_filename: str,
    file_type: str,
    row_count: int,
    column_mapping: Dict[str, str],
    added_columns: List[str],
    libraries: List[Dict[str, str]],
    timestamp: Optional[datetime] = None,
    reproducibility: Optional[Dict[str, Any]] = None,
    summary: Optional[Dict[str, Any]] = None,
    sampling: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a complete ``run_manifest.json`` dict.

    Parameters
    ----------
    tool_key:
        Registry key (e.g. ``"fa_g2p"``, ``"stratified"``).
    language:
        ISO language code or ``None`` for language-agnostic tools.
    timestamp:
        If ``None``, uses ``utc_now()``.
    """
    ts = timestamp or utc_now()
    rkey = registry_key(language, tool_key) if language else tool_key
    spec = TOOL_REGISTRY.get(rkey)
    manifest_tool = spec.manifest_name if spec else tool_key

    manifest: Dict[str, Any] = {
        "lexprep_version": __version__,
        "timestamp_utc": ts.isoformat(),
        "tool": manifest_tool,
        "citation": CITATION_INFO,
        "input": {
            "original_filename": original_filename,
            "file_type": file_type,
            "row_count": row_count,
            "column_mapping": column_mapping,
        },
        "pipeline": {
            "added_columns": added_columns,
        },
    }

    # Only include language for language-specific tools
    if language:
        manifest["language"] = language

    # Only include libraries when non-empty
    if libraries:
        manifest["pipeline"]["libraries"] = libraries

    if reproducibility:
        manifest["reproducibility"] = reproducibility
    if summary:
        manifest["summary"] = summary
    if sampling:
        manifest["sampling"] = sampling

    return manifest


# ---------------------------------------------------------------------------
# Manifest validation (for tests)
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = {
    "lexprep_version",
    "timestamp_utc",
    "tool",
    "citation",
    "input",
    "pipeline",
}
_REQUIRED_INPUT_KEYS = {"original_filename", "file_type", "row_count", "column_mapping"}
_REQUIRED_PIPELINE_KEYS = {"added_columns"}


def validate_manifest(manifest: Dict[str, Any]) -> None:
    """Raise ``ValueError`` if the manifest is structurally invalid."""
    missing = _REQUIRED_KEYS - set(manifest.keys())
    if missing:
        raise ValueError(f"Missing top-level keys: {missing}")

    missing_input = _REQUIRED_INPUT_KEYS - set(manifest.get("input", {}).keys())
    if missing_input:
        raise ValueError(f"Missing input keys: {missing_input}")

    missing_pipeline = _REQUIRED_PIPELINE_KEYS - set(manifest.get("pipeline", {}).keys())
    if missing_pipeline:
        raise ValueError(f"Missing pipeline keys: {missing_pipeline}")

    if not isinstance(manifest["input"]["row_count"], int):
        raise ValueError("input.row_count must be an integer")
