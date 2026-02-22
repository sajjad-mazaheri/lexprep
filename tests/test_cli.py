"""Tests for CLI commands (lexprep.cli).

Tests the Typer CLI by invoking commands with CliRunner.
NLP-heavy commands are conditionally skipped.
All commands now output ZIP files containing run_manifest.json + enriched/sample files.
"""

import json
import zipfile
from io import BytesIO

import pandas as pd
import pytest
from typer.testing import CliRunner

from lexprep.cli import app

runner = CliRunner()


def _make_csv(tmp_path, name="input.csv", words=None, extra_cols=None):
    """Create a simple CSV test file."""
    if words is None:
        words = ["hello", "world", "test"]
    data = {"word": words}
    if extra_cols:
        data.update(extra_cols)
    df = pd.DataFrame(data)
    p = tmp_path / name
    df.to_csv(p, index=False)
    return str(p)


def _read_zip_enriched(zip_path, fmt="csv"):
    """Read the enriched file from a ZIP output."""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        # Find the enriched or sample file
        enriched = [n for n in names if "__enriched." in n]
        sample = [n for n in names if "__sample." in n]
        target = enriched[0] if enriched else (sample[0] if sample else None)
        assert target is not None, f"No enriched/sample file in ZIP: {names}"
        data = zf.read(target)
        if target.endswith(".csv"):
            return pd.read_csv(BytesIO(data))
        elif target.endswith(".xlsx"):
            return pd.read_excel(BytesIO(data))
        elif target.endswith(".tsv"):
            return pd.read_csv(BytesIO(data), sep="\t")
    return None


def _read_zip_manifest(zip_path):
    """Read run_manifest.json from a ZIP output."""
    with zipfile.ZipFile(zip_path) as zf:
        return json.loads(zf.read("run_manifest.json"))


class TestCLIGeneral:
    def test_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "lexprep" in result.stdout.lower()

    def test_fa_help(self):
        result = runner.invoke(app, ["fa", "--help"])
        assert result.exit_code == 0

    def test_en_help(self):
        result = runner.invoke(app, ["en", "--help"])
        assert result.exit_code == 0

    def test_ja_help(self):
        result = runner.invoke(app, ["ja", "--help"])
        assert result.exit_code == 0

    def test_sample_help(self):
        result = runner.invoke(app, ["sample", "--help"])
        assert result.exit_code == 0

    def test_length_help(self):
        result = runner.invoke(app, ["length", "--help"])
        assert result.exit_code == 0


class TestLengthCLI:
    def test_length_basic(self, tmp_path):
        inp = _make_csv(tmp_path)
        out = str(tmp_path / "output.csv")
        result = runner.invoke(app, ["length", inp, out])
        assert result.exit_code == 0
        zip_path = tmp_path / "output.zip"
        assert zip_path.exists()
        df = _read_zip_enriched(zip_path)
        assert "length_chars" in df.columns
        assert len(df) == 3
        # "hello" = 5, "world" = 5, "test" = 4
        assert list(df["length_chars"]) == [5, 5, 4]

    def test_length_manifest(self, tmp_path):
        inp = _make_csv(tmp_path)
        out = str(tmp_path / "output.csv")
        runner.invoke(app, ["length", inp, out])
        m = _read_zip_manifest(tmp_path / "output.zip")
        assert m["tool"] == "length"
        assert "length_distribution" in m.get("summary", {})
        assert m["summary"]["length_method"] == "unicode_codepoints"


class TestEnglishCLI:
    def test_en_syllables(self, tmp_path):
        pytest.importorskip("pyphen")
        inp = _make_csv(tmp_path)
        out = str(tmp_path / "output.csv")
        result = runner.invoke(app, ["en", "syllables", inp, out])
        assert result.exit_code == 0
        zip_path = tmp_path / "output.zip"
        df = _read_zip_enriched(zip_path)
        assert "syllables" in df.columns
        assert len(df) == 3

    def test_en_g2p(self, tmp_path):
        pytest.importorskip("g2p_en")
        inp = _make_csv(tmp_path)
        out = str(tmp_path / "output.csv")
        result = runner.invoke(app, ["en", "g2p", inp, out])
        assert result.exit_code == 0
        df = _read_zip_enriched(tmp_path / "output.zip")
        assert "pronunciation" in df.columns

    def test_en_pos(self, tmp_path):
        spacy = pytest.importorskip("spacy")
        try:
            spacy.load("en_core_web_sm")
        except OSError:
            pytest.skip("spaCy model en_core_web_sm not available")
        inp = _make_csv(tmp_path)
        out = str(tmp_path / "output.csv")
        result = runner.invoke(app, ["en", "pos", inp, out])
        assert result.exit_code == 0
        df = _read_zip_enriched(tmp_path / "output.zip")
        assert "pos" in df.columns
        assert "tag" in df.columns
        assert "lemma" in df.columns


class TestPersianCLI:
    def test_fa_g2p(self, tmp_path):
        pytest.importorskip("PersianG2p")
        inp = _make_csv(tmp_path, words=["سلام", "کتاب", "مادر"])
        out = str(tmp_path / "output.csv")
        result = runner.invoke(app, ["fa", "g2p", inp, out])
        assert result.exit_code == 0
        df = _read_zip_enriched(tmp_path / "output.zip")
        assert "pronunciation" in df.columns
        assert len(df) == 3

    def test_fa_syllables_orthographic(self, tmp_path):
        inp = _make_csv(tmp_path, words=["سلام", "کتاب"])
        out = str(tmp_path / "output.csv")
        result = runner.invoke(app, ["fa", "syllables", inp, out])
        assert result.exit_code == 0
        df = _read_zip_enriched(tmp_path / "output.zip")
        assert "syllabified" in df.columns
        assert "syllables" in df.columns

    def test_fa_syllables_with_g2p(self, tmp_path):
        pytest.importorskip("PersianG2p")
        inp = _make_csv(tmp_path, words=["سلام", "کتاب"])
        out = str(tmp_path / "output.csv")
        result = runner.invoke(app, ["fa", "syllables", inp, out, "--with-g2p"])
        assert result.exit_code == 0
        df = _read_zip_enriched(tmp_path / "output.zip")
        assert "pronunciation" in df.columns
        assert "syllables" in df.columns

    def test_fa_pos(self, tmp_path):
        pytest.importorskip("stanza")
        inp = _make_csv(tmp_path, words=["سلام", "کتاب", "زیبا"])
        out = str(tmp_path / "output.csv")
        result = runner.invoke(app, ["fa", "pos", inp, out])
        assert result.exit_code == 0
        df = _read_zip_enriched(tmp_path / "output.zip")
        assert "pos" in df.columns
        assert "lemma" in df.columns
        assert len(df) == 3


class TestJapaneseCLI:
    def test_ja_pos_unidic(self, tmp_path):
        pytest.importorskip("fugashi")
        pytest.importorskip("unidic_lite")
        inp = _make_csv(tmp_path, words=["本", "食べる", "猫"])
        out = str(tmp_path / "output.csv")
        result = runner.invoke(app, ["ja", "pos", inp, out])
        assert result.exit_code == 0
        df = _read_zip_enriched(tmp_path / "output.zip")
        assert "pos" in df.columns
        assert "pos_english" in df.columns
        assert "lemma" in df.columns
        assert len(df) == 3

    def test_ja_pos_stanza(self, tmp_path):
        pytest.importorskip("stanza")
        inp = _make_csv(tmp_path, words=["本", "食べる", "猫"])
        out = str(tmp_path / "output.csv")
        result = runner.invoke(app, ["ja", "pos", inp, out, "--method", "stanza"])
        assert result.exit_code == 0
        df = _read_zip_enriched(tmp_path / "output.zip")
        assert "pos" in df.columns
        assert len(df) == 3


class TestSamplingCLI:
    def test_stratified(self, tmp_path):
        inp = _make_csv(
            tmp_path, words=[f"w{i}" for i in range(50)], extra_cols={"score": list(range(50))}
        )
        out = str(tmp_path / "sampled.csv")
        result = runner.invoke(
            app, ["sample", "stratified", inp, out, "--score", "score", "--n", "15", "--bins", "3"]
        )
        assert result.exit_code == 0
        zip_path = tmp_path / "sampled.zip"
        assert zip_path.exists()

        # Verify ZIP structure
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            assert "run_manifest.json" in names
            assert any("__sample." in n for n in names)
            assert any("__excluded." in n for n in names)
            assert "sampling_audit.xlsx" in names

        # Read and verify sample
        df = _read_zip_enriched(zip_path)
        assert len(df) == 15
        assert "bin_id" in df.columns

    def test_shuffle(self, tmp_path):
        f1 = _make_csv(tmp_path, name="f1.csv", words=["a", "b", "c"], extra_cols={"x": [1, 2, 3]})
        f2 = _make_csv(tmp_path, name="f2.csv", words=["d", "e", "f"], extra_cols={"x": [4, 5, 6]})
        out_dir = str(tmp_path / "shuffled")
        result = runner.invoke(app, ["sample", "shuffle", f1, f2, out_dir, "--seed", "42"])
        assert result.exit_code == 0

        # Find the ZIP file in the output directory
        shuffled_dir = tmp_path / "shuffled"
        zips = list(shuffled_dir.glob("*.zip"))
        assert len(zips) == 1

        # Verify ZIP has manifest + shuffled files
        with zipfile.ZipFile(zips[0]) as zf:
            names = zf.namelist()
            assert "run_manifest.json" in names
            shuffled_files = [n for n in names if "_shuffled." in n]
            assert len(shuffled_files) == 2

    def test_shuffle_too_few_files(self, tmp_path):
        f1 = _make_csv(tmp_path, name="f1.csv")
        out_dir = str(tmp_path / "shuffled")
        result = runner.invoke(app, ["sample", "shuffle", f1, out_dir])
        assert result.exit_code != 0 or "at least 2" in (result.stdout + result.output).lower()
