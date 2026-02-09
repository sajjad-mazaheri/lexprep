""" tests for Persian syllable counting (lexprep.fa.syllables)."""

from lexprep.fa.syllables import (
    DEFAULT_VOWEL_PATTERNS,
    count_syllables_from_pronunciation,
    syllabify_orthographic,
)


# count_syllables_from_pronunciation


class TestPhoneticSyllableCount:
    def test_two_syllables(self):
        assert count_syllables_from_pronunciation("mā dar") == 2

    def test_empty_string(self):
        assert count_syllables_from_pronunciation("") == 0

    def test_none_input(self):
        assert count_syllables_from_pronunciation(None) == 0

    def test_single_vowel(self):
        assert count_syllables_from_pronunciation("ā") == 1

    def test_no_vowels(self):
        assert count_syllables_from_pronunciation("brk") == 0

    def test_spaces_ignored_in_count(self):
        # Spaces are removed before counting.
        assert count_syllables_from_pronunciation("sa lam") == count_syllables_from_pronunciation("salam")

    def test_long_vowels(self):
        # Long vowels count once each.
        assert count_syllables_from_pronunciation("āīū") == 3

    def test_custom_patterns(self):
        result = count_syllables_from_pronunciation("xaya", vowel_patterns=["a"])
        assert result == 2


# syllabify_orthographic


class TestOrthographicSyllabify:
    def test_basic_word(self):
        syllabified, count = syllabify_orthographic("کتاب")
        assert isinstance(syllabified, str)
        assert isinstance(count, int)
        assert count >= 1

    def test_empty_string(self):
        syllabified, count = syllabify_orthographic("")
        assert syllabified == ""
        assert count == 0

    def test_whitespace_only(self):
        syllabified, count = syllabify_orthographic("   ")
        assert syllabified == ""
        assert count == 0

    def test_delimiter_in_output(self):
        syllabified, count = syllabify_orthographic("کتاب", delimiter="-")
        if count > 1:
            assert "-" in syllabified

    def test_custom_delimiter(self):
        syllabified, count = syllabify_orthographic("کتاب", delimiter=".")
        if count > 1:
            assert "." in syllabified

    def test_single_char(self):
        _, count = syllabify_orthographic("ا")
        assert count >= 1
