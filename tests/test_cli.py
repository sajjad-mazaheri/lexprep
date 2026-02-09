"""Tests for CLI commands (lexprep.cli).

Tests the Typer CLI by invoking commands with CliRunner.
NLP-heavy commands are conditionally skipped.
"""

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


class TestEnglishCLI:
    def test_en_syllables(self, tmp_path):
        pytest.importorskip("pyphen")
        inp = _make_csv(tmp_path)
        out = str(tmp_path / "output.csv")
        result = runner.invoke(app, ["en", "syllables", inp, out])
        assert result.exit_code == 0
        df = pd.read_csv(out)
        assert "syllables" in df.columns
        assert len(df) == 3

    def test_en_g2p(self, tmp_path):
        pytest.importorskip("g2p_en")
        inp = _make_csv(tmp_path)
        out = str(tmp_path / "output.csv")
        result = runner.invoke(app, ["en", "g2p", inp, out])
        assert result.exit_code == 0
        df = pd.read_csv(out)
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
        df = pd.read_csv(out)
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
        df = pd.read_csv(out)
        assert "pronunciation" in df.columns
        assert len(df) == 3

    def test_fa_syllables_orthographic(self, tmp_path):
        inp = _make_csv(tmp_path, words=["سلام", "کتاب"])
        out = str(tmp_path / "output.csv")
        result = runner.invoke(app, ["fa", "syllables", inp, out])
        assert result.exit_code == 0
        df = pd.read_csv(out)
        assert "syllabified" in df.columns
        assert "syllables" in df.columns

    def test_fa_syllables_with_g2p(self, tmp_path):
        pytest.importorskip("PersianG2p")
        inp = _make_csv(tmp_path, words=["سلام", "کتاب"])
        out = str(tmp_path / "output.csv")
        result = runner.invoke(app, ["fa", "syllables", inp, out, "--with-g2p"])
        assert result.exit_code == 0
        df = pd.read_csv(out)
        assert "pronunciation" in df.columns
        assert "syllables" in df.columns

    def test_fa_pos(self, tmp_path):
        pytest.importorskip("stanza")
        inp = _make_csv(tmp_path, words=["سلام", "کتاب", "زیبا"])
        out = str(tmp_path / "output.csv")
        result = runner.invoke(app, ["fa", "pos", inp, out])
        assert result.exit_code == 0
        df = pd.read_csv(out)
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
        df = pd.read_csv(out)
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
        df = pd.read_csv(out)
        assert "pos" in df.columns
        assert len(df) == 3


class TestSamplingCLI:
    def test_stratified(self, tmp_path):
        inp = _make_csv(tmp_path, words=[f"w{i}" for i in range(50)],
                        extra_cols={"score": list(range(50))})
        out = str(tmp_path / "sampled.csv")
        result = runner.invoke(app, [
            "sample", "stratified", inp, out,
            "--score", "score", "--n", "15", "--bins", "3"
        ])
        assert result.exit_code == 0
        df = pd.read_csv(out)
        assert len(df) == 15

    def test_shuffle(self, tmp_path):
        f1 = _make_csv(tmp_path, name="f1.csv", words=["a", "b", "c"],
                       extra_cols={"x": [1, 2, 3]})
        f2 = _make_csv(tmp_path, name="f2.csv", words=["d", "e", "f"],
                       extra_cols={"x": [4, 5, 6]})
        out_dir = str(tmp_path / "shuffled")
        result = runner.invoke(app, ["sample", "shuffle", f1, f2, out_dir, "--seed", "42"])
        assert result.exit_code == 0

    def test_shuffle_too_few_files(self, tmp_path):
        f1 = _make_csv(tmp_path, name="f1.csv")
        out_dir = str(tmp_path / "shuffled")
        result = runner.invoke(app, ["sample", "shuffle", f1, out_dir])
        assert result.exit_code != 0 or "at least 2" in (result.stdout + result.output).lower()
