"""Tests for English syllable counting (lexprep.en.syllables).

require pyphen: pip install 'lexprep[en]'
"""

import pytest

try:
    from lexprep.en.syllables import count_syllables, count_syllables_batch
    _HAS_EN = True
except ImportError:
    _HAS_EN = False

pytestmark = pytest.mark.skipif(
    not _HAS_EN, reason="English deps not installed",
)


class TestCountSyllables:
    def test_monosyllabic(self):
        assert count_syllables("cat") == 1

    def test_bisyllabic(self):
        assert count_syllables("water") == 2

    def test_trisyllabic(self):
        assert count_syllables("beautiful") >= 3

    def test_empty_string(self):
        assert count_syllables("") == 0

    def test_whitespace(self):
        assert count_syllables("   ") == 0

    def test_numeric_coerced(self):
        # word gets converted to str
        result = count_syllables("123")
        assert isinstance(result, int)


class TestCountSyllablesBatch:
    def test_batch(self):
        words = ["cat", "water", "beautiful"]
        results = count_syllables_batch(words)
        assert len(results) == 3
        assert results[0] == 1
        assert results[1] == 2
        assert results[2] >= 3

    def test_empty_list(self):
        assert count_syllables_batch([]) == []
