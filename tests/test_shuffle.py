"""Tests for lexprep.sampling.shuffle_rows module."""

import pandas as pd
import pytest

from lexprep.sampling.shuffle_rows import shuffle_corresponding_rows


@pytest.fixture
def two_dfs():
    """Two DataFrames with identical shape."""
    df1 = pd.DataFrame({"word": ["a", "b", "c"], "score": [1, 2, 3]})
    df2 = pd.DataFrame({"word": ["x", "y", "z"], "score": [10, 20, 30]})
    return [df1, df2]


class TestShuffleCorrespondingRows:
    def test_output_count(self, two_dfs):
        out, report = shuffle_corresponding_rows(two_dfs, seed=1)
        assert len(out) == 2
        assert report.n_files == 2
        assert report.n_rows == 3

    def test_preserves_row_count(self, two_dfs):
        out, _ = shuffle_corresponding_rows(two_dfs, seed=1)
        for df in out:
            assert len(df) == 3

    def test_seed_reproducibility(self, two_dfs):
        out1, _ = shuffle_corresponding_rows(two_dfs, seed=42)
        out2, _ = shuffle_corresponding_rows(two_dfs, seed=42)
        for d1, d2 in zip(out1, out2):
            assert d1.values.tolist() == d2.values.tolist()

    def test_different_seeds_differ(self, two_dfs):
        out1, _ = shuffle_corresponding_rows(two_dfs, seed=1)
        out2, _ = shuffle_corresponding_rows(two_dfs, seed=999)
        # At least one output should differ
        any_different = False
        for d1, d2 in zip(out1, out2):
            if d1.values.tolist() != d2.values.tolist():
                any_different = True
        assert any_different

    def test_fewer_than_two_files_raises(self):
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ValueError, match="at least 2"):
            shuffle_corresponding_rows([df])

    def test_mismatched_rows_raises(self):
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="Row count mismatch"):
            shuffle_corresponding_rows([df1, df2])

    def test_n_columns_limits_output(self):
        df1 = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        df2 = pd.DataFrame({"a": [4], "b": [5], "c": [6]})
        out, report = shuffle_corresponding_rows([df1, df2], n_columns=2)
        assert list(out[0].columns) == ["a", "b"]
        assert report.used_columns == ["a", "b"]

    def test_three_files(self):
        dfs = [
            pd.DataFrame({"x": [1, 2]}),
            pd.DataFrame({"x": [3, 4]}),
            pd.DataFrame({"x": [5, 6]}),
        ]
        out, report = shuffle_corresponding_rows(dfs, seed=7)
        assert report.n_files == 3
        assert len(out) == 3
