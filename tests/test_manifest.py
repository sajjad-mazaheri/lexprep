"""Tests for lexprep.manifest module (D2)."""

from datetime import datetime, timezone

import pytest

from lexprep.manifest import (
    CITATION_INFO,
    TOOL_REGISTRY,
    build_manifest,
    format_timestamp_filename,
    get_library_version,
    registry_key,
    sanitize_basename,
    utc_now,
    validate_manifest,
)


class TestToolRegistry:
    def test_all_keys_have_manifest_name(self):
        for key, spec in TOOL_REGISTRY.items():
            assert spec.manifest_name, f"No manifest_name for {key}"

    def test_registry_key_with_language(self):
        assert registry_key("fa", "g2p") == "fa_g2p"
        assert registry_key("en", "pos") == "en_pos"

    def test_registry_key_without_language(self):
        assert registry_key(None, "stratified") == "stratified"
        assert registry_key(None, "shuffle") == "shuffle"

    def test_registry_key_fallback(self):
        assert registry_key("fa", "length") == "length"


class TestTimestamp:
    def test_utc_now_is_utc(self):
        ts = utc_now()
        assert ts.tzinfo == timezone.utc

    def test_format_timestamp_filename(self):
        ts = datetime(2026, 2, 20, 14, 30, 0, tzinfo=timezone.utc)
        assert format_timestamp_filename(ts) == "20260220T143000Z"


class TestSanitizeBasename:
    def test_normal_name(self):
        assert sanitize_basename("wordlist") == "wordlist"

    def test_spaces(self):
        assert sanitize_basename("my file") == "my_file"

    def test_special_chars(self):
        assert sanitize_basename('file<>:"/name') == "file_name"

    def test_empty(self):
        assert sanitize_basename("") == "output"


class TestBuildManifest:
    def _make_manifest(self, **kwargs):
        defaults = {
            "tool_key": "g2p",
            "language": "fa",
            "original_filename": "test.xlsx",
            "file_type": "xlsx",
            "row_count": 100,
            "column_mapping": {"word_column": "word"},
            "added_columns": ["pronunciation"],
            "libraries": [{"name": "PersianG2p", "version": "0.1.5"}],
        }
        defaults.update(kwargs)
        return build_manifest(**defaults)

    def test_required_fields_present(self):
        m = self._make_manifest()
        validate_manifest(m)  # Should not raise

    def test_tool_name_mapped(self):
        m = self._make_manifest(tool_key="g2p", language="fa")
        assert m["tool"] == "g2p"

    def test_syllable_tool_mapped(self):
        m = self._make_manifest(tool_key="syllables", language="en")
        assert m["tool"] == "syllable_count"

    def test_pos_tool_mapped(self):
        m = self._make_manifest(tool_key="pos", language="fa")
        assert m["tool"] == "pos_tagging"

    def test_version_present(self):
        m = self._make_manifest()
        assert m["lexprep_version"] == "1.0.0"

    def test_timestamp_utc(self):
        m = self._make_manifest()
        assert "T" in m["timestamp_utc"]

    def test_citation_present(self):
        m = self._make_manifest()
        assert m["citation"]["doi"] == "10.5281/zenodo.18713755"

    def test_input_section(self):
        m = self._make_manifest()
        assert m["input"]["original_filename"] == "test.xlsx"
        assert m["input"]["row_count"] == 100

    def test_optional_fields_omitted_when_none(self):
        m = self._make_manifest()
        assert "reproducibility" not in m
        assert "summary" not in m
        assert "sampling" not in m

    def test_optional_fields_included_when_set(self):
        m = self._make_manifest(
            reproducibility={"seed": 42},
            summary={"pos_distribution": {"NOUN": 10}},
        )
        assert m["reproducibility"]["seed"] == 42
        assert "pos_distribution" in m["summary"]


class TestValidateManifest:
    def test_valid(self):
        m = build_manifest(
            tool_key="g2p",
            language="fa",
            original_filename="test.xlsx",
            file_type="xlsx",
            row_count=10,
            column_mapping={},
            added_columns=[],
            libraries=[],
        )
        validate_manifest(m)  # Should not raise

    def test_missing_top_level(self):
        with pytest.raises(ValueError, match="Missing top-level"):
            validate_manifest({"tool": "g2p"})

    def test_row_count_must_be_int(self):
        m = build_manifest(
            tool_key="g2p",
            language="fa",
            original_filename="t.xlsx",
            file_type="xlsx",
            row_count=10,
            column_mapping={},
            added_columns=[],
            libraries=[],
        )
        m["input"]["row_count"] = "10"
        with pytest.raises(ValueError, match="row_count"):
            validate_manifest(m)


class TestCitationSync:
    """Ensure hardcoded CITATION_INFO matches CITATION.cff."""

    def test_doi_matches_cff(self):
        import pathlib

        cff_path = pathlib.Path(__file__).parent.parent / "CITATION.cff"
        if not cff_path.exists():
            pytest.skip("CITATION.cff not found")
        content = cff_path.read_text(encoding="utf-8")
        # Check DOI is present in the CFF file
        assert CITATION_INFO["doi"] in content

    def test_url_matches_cff(self):
        import pathlib

        cff_path = pathlib.Path(__file__).parent.parent / "CITATION.cff"
        if not cff_path.exists():
            pytest.skip("CITATION.cff not found")
        content = cff_path.read_text(encoding="utf-8")
        assert CITATION_INFO["url"] in content


class TestGetLibraryVersion:
    def test_known_package(self):
        ver = get_library_version("pandas")
        assert ver != "unknown"

    def test_unknown_package(self):
        ver = get_library_version("nonexistent_package_12345")
        assert ver == "unknown"
