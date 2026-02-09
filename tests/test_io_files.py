"""Test read/write/wordlist utilities."""

import pandas as pd
import pytest

from lexprep.io.files import (
    _guess_word_column,
    coerce_str_series,
    ensure_column,
    read_table,
    read_wordlist,
    write_table,
)

# _guess_word_column tests


class TestGuessWordColumn:
    def test_finds_word(self):
        df = pd.DataFrame({"word": ["a"], "other": [1]})
        assert _guess_word_column(df) == "word"

    def test_finds_item_column(self):
        df = pd.DataFrame({"Item": ["a"], "freq": [1]})
        assert _guess_word_column(df) == "Item"

    def test_fallback_first_column(self):
        df = pd.DataFrame({"xyz": ["a"], "abc": [1]})
        assert _guess_word_column(df) == "xyz"

    def test_empty_columns_raises(self):
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="no columns"):
            _guess_word_column(df)


# read_table / write_table


class TestReadWriteTable:
    def test_csv_roundtrip(self, tmp_path):
        p = tmp_path / "data.csv"
        df = pd.DataFrame({"word": ["hello", "world"], "score": [1, 2]})
        write_table(df, p)
        loaded = read_table(p)
        assert list(loaded.columns) == ["word", "score"]
        assert len(loaded) == 2
        assert loaded["word"].tolist() == ["hello", "world"]

    def test_tsv_roundtrip(self, tmp_path):
        p = tmp_path / "data.tsv"
        df = pd.DataFrame({"w": ["a", "b"]})
        write_table(df, p)
        loaded = read_table(p)
        assert loaded["w"].tolist() == ["a", "b"]

    def test_xlsx_roundtrip(self, tmp_path):
        p = tmp_path / "data.xlsx"
        df = pd.DataFrame({"col": [10, 20, 30]})
        write_table(df, p)
        loaded = read_table(p)
        assert loaded["col"].tolist() == [10, 20, 30]

    def test_unsupported_ext_read(self, tmp_path):
        p = tmp_path / "data.json"
        p.write_text("{}")
        with pytest.raises(ValueError, match="Unsupported tabular file"):
            read_table(p)

    def test_unsupported_ext_write(self, tmp_path):
        p = tmp_path / "data.json"
        with pytest.raises(ValueError, match="Unsupported output format"):
            write_table(pd.DataFrame({"a": [1]}), p)


# read_wordlist


class TestReadWordlist:
    def test_txt_file(self, tmp_path):
        p = tmp_path / "words.txt"
        p.write_text("apple\nbanana\ncherry\n", encoding="utf-8")
        df, col = read_wordlist(p)
        assert col == "word"
        assert len(df) == 3
        assert df["word"].tolist() == ["apple", "banana", "cherry"]

    def test_txt_skips_blank_lines(self, tmp_path):
        p = tmp_path / "words.txt"
        p.write_text("apple\n\n\nbanana\n", encoding="utf-8")
        df, col = read_wordlist(p)
        assert len(df) == 2

    def test_csv_auto_guess(self, tmp_path):
        p = tmp_path / "data.csv"
        pd.DataFrame({"word": ["a", "b"], "freq": [5, 3]}).to_csv(p, index=False)
        df, col = read_wordlist(p)
        assert col == "word"

    def test_csv_explicit_col(self, tmp_path):
        p = tmp_path / "data.csv"
        pd.DataFrame({"mywords": ["x"], "val": [1]}).to_csv(p, index=False)
        df, col = read_wordlist(p, word_col="mywords")
        assert col == "mywords"

    def test_missing_column_raises(self, tmp_path):
        p = tmp_path / "data.csv"
        pd.DataFrame({"a": [1]}).to_csv(p, index=False)
        with pytest.raises(ValueError, match="not found"):
            read_wordlist(p, word_col="missing")


# helper utilities


class TestHelpers:
    def test_coerce_str_series(self):
        s = pd.Series([1, "  hi  ", 3.14])
        result = coerce_str_series(s)
        assert result.tolist() == ["1", "hi", "3.14"]

    def test_ensure_column_adds(self):
        df = pd.DataFrame({"a": [1]})
        df = ensure_column(df, "new_col")
        assert "new_col" in df.columns
        assert df["new_col"].iloc[0] == ""

    def test_ensure_column_no_overwrite(self):
        df = pd.DataFrame({"a": [42]})
        df = ensure_column(df, "a")
        assert df["a"].iloc[0] == 42
