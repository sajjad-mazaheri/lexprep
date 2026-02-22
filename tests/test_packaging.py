"""Tests for lexprep.packaging module (D1)."""

import json
import zipfile
from datetime import datetime, timezone
from io import BytesIO

import pandas as pd

from lexprep.manifest import build_manifest
from lexprep.packaging import build_zip, make_output_filename, make_zip_filename


class TestMakeZipFilename:
    def test_basic(self):
        ts = datetime(2026, 2, 20, 14, 30, 0, tzinfo=timezone.utc)
        name = make_zip_filename("wordlist", "g2p", "fa", ts)
        assert name == "wordlist__g2p__fa__20260220T143000Z.zip"

    def test_no_language(self):
        ts = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        name = make_zip_filename("data", "row_shuffle", None, ts)
        assert name == "data__row_shuffle__all__20260101T000000Z.zip"

    def test_sanitized_basename(self):
        ts = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        name = make_zip_filename("my file", "g2p", "en", ts)
        assert "my_file" in name
        assert " " not in name


class TestMakeOutputFilename:
    def test_enriched(self):
        assert make_output_filename("wordlist", "xlsx") == "wordlist__enriched.xlsx"

    def test_sample(self):
        assert make_output_filename("data", "xlsx", is_sampling=True) == "data__sample.xlsx"

    def test_csv(self):
        assert make_output_filename("words", "csv") == "words__enriched.csv"


class TestBuildZip:
    def _make_manifest(self, tool="g2p", language="fa"):
        return build_manifest(
            tool_key=tool,
            language=language,
            original_filename="test.xlsx",
            file_type="xlsx",
            row_count=3,
            column_mapping={"word_column": "word"},
            added_columns=["pronunciation"],
            libraries=[],
        )

    def test_basic_zip_structure(self):
        manifest = self._make_manifest()
        df = pd.DataFrame({"word": ["a", "b"], "pronunciation": ["x", "y"]})
        zip_bytes, zip_name = build_zip(
            manifest=manifest,
            main_df=df,
            input_basename="test",
            output_ext="xlsx",
        )
        assert zip_name.endswith(".zip")
        assert "__g2p__fa__" in zip_name

        with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
            assert "run_manifest.json" in names
            assert "test__enriched.xlsx" in names
            assert len(names) == 2

    def test_manifest_is_valid_json(self):
        manifest = self._make_manifest()
        df = pd.DataFrame({"word": ["a"]})
        zip_bytes, _ = build_zip(
            manifest=manifest,
            main_df=df,
            input_basename="test",
            output_ext="xlsx",
        )
        with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
            data = json.loads(zf.read("run_manifest.json"))
            assert data["tool"] == "g2p"
            assert data["language"] == "fa"

    def test_sampling_zip_structure(self):
        manifest = build_manifest(
            tool_key="stratified",
            language=None,
            original_filename="data.xlsx",
            file_type="xlsx",
            row_count=100,
            column_mapping={"score_column": "freq"},
            added_columns=["bin_id"],
            libraries=[],
        )
        sample_df = pd.DataFrame({"word": ["a"], "bin_id": [0]})
        excluded_df = pd.DataFrame({"word": ["b"], "bin_id": [1]})
        audit_bytes = b"fake audit data"

        zip_bytes, zip_name = build_zip(
            manifest=manifest,
            main_df=sample_df,
            input_basename="data",
            output_ext="xlsx",
            is_sampling=True,
            excluded_df=excluded_df,
            audit_bytes=audit_bytes,
        )

        with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
            assert "run_manifest.json" in names
            assert "data__sample.xlsx" in names
            assert "sampling_audit.xlsx" in names
            assert "data__excluded.xlsx" in names
            assert len(names) == 4

    def test_extra_files_sorted(self):
        manifest = self._make_manifest(tool="shuffle", language=None)
        manifest["tool"] = "row_shuffle"  # Override since language=None
        zip_bytes, _ = build_zip(
            manifest=manifest,
            main_df=pd.DataFrame(),
            input_basename="shuffle",
            output_ext="xlsx",
            extra_files=[("b_shuffled.xlsx", b"b"), ("a_shuffled.xlsx", b"a")],
        )
        with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
            # Extra files should be sorted
            extra = [n for n in names if n.endswith("_shuffled.xlsx")]
            assert extra == ["a_shuffled.xlsx", "b_shuffled.xlsx"]

    def test_output_filename_pattern_enriched(self):
        manifest = self._make_manifest()
        df = pd.DataFrame({"word": ["a"]})
        zip_bytes, _ = build_zip(
            manifest=manifest,
            main_df=df,
            input_basename="mywords",
            output_ext="csv",
        )
        with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
            assert "mywords__enriched.csv" in zf.namelist()

    def test_output_filename_pattern_sample(self):
        manifest = self._make_manifest()
        df = pd.DataFrame({"word": ["a"]})
        zip_bytes, _ = build_zip(
            manifest=manifest,
            main_df=df,
            input_basename="mydata",
            output_ext="xlsx",
            is_sampling=True,
        )
        with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
            assert "mydata__sample.xlsx" in zf.namelist()

    def test_deterministic_file_order(self):
        """ZIP should have files in a consistent order."""
        manifest = self._make_manifest()
        df = pd.DataFrame({"word": ["a"]})
        zip_bytes, _ = build_zip(
            manifest=manifest,
            main_df=df,
            input_basename="test",
            output_ext="xlsx",
        )
        with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
            assert names[0] == "run_manifest.json"
            assert names[1] == "test__enriched.xlsx"
