from lexprep.fa.syllables import count_syllables_from_pronunciation, syllabify_orthographic


def test_phonetic_syllable_count_basic():
    assert count_syllables_from_pronunciation("mā dar") == 2
    assert count_syllables_from_pronunciation("salam") >= 1
    assert count_syllables_from_pronunciation("") == 0


def test_orthographic_syllabify_returns_count():
    s, c = syllabify_orthographic("کتاب")
    assert isinstance(s, str)
    assert isinstance(c, int)
    assert c >= 1
