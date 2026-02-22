"""Tests for lexprep.length module (D3)."""

from lexprep.length import (
    LENGTH_METHOD,
    LengthDistribution,
    compute_length_chars,
    length_distribution,
)


class TestComputeLengthChars:
    def test_ascii(self):
        assert compute_length_chars(["hello", "world"]) == [5, 5]

    def test_persian(self):
        # «سلام» = 4 characters
        assert compute_length_chars(["سلام"]) == [4]

    def test_japanese(self):
        # «東京» = 2 characters
        assert compute_length_chars(["東京"]) == [2]

    def test_empty_string(self):
        assert compute_length_chars([""]) == [0]

    def test_whitespace_only(self):
        assert compute_length_chars(["   "]) == [0]

    def test_mixed(self):
        result = compute_length_chars(["abc", "", "سلام", "  "])
        assert result == [3, 0, 4, 0]

    def test_unicode_emoji(self):
        # Emoji is 1 codepoint
        assert compute_length_chars(["😀"]) == [1]


class TestLengthDistribution:
    def test_basic(self):
        dist = length_distribution([3, 5, 7])
        assert dist is not None
        assert dist.min == 3
        assert dist.max == 7
        assert dist.mean == 5.0
        assert dist.median == 5.0

    def test_all_zeros(self):
        assert length_distribution([0, 0, 0]) is None

    def test_empty_list(self):
        assert length_distribution([]) is None

    def test_single_value(self):
        dist = length_distribution([4])
        assert dist.min == 4
        assert dist.max == 4

    def test_with_zeros_filtered(self):
        dist = length_distribution([0, 3, 0, 5])
        assert dist is not None
        assert dist.min == 3
        assert dist.max == 5

    def test_to_dict(self):
        dist = LengthDistribution(min=1, max=10, mean=5.0, median=5.0)
        d = dist.to_dict()
        assert d == {"min": 1, "max": 10, "mean": 5.0, "median": 5.0}


class TestLengthMethod:
    def test_method_string(self):
        assert LENGTH_METHOD == "unicode_codepoints"
