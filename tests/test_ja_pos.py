"""Tests for Japanese POS tagging (UniDic + Stanza).

Requires: pip install 'lexprep[ja]'
"""

import pytest

# UniDic

try:
    from lexprep.ja.pos_unidic import UniDicResult, tag_with_unidic
    _HAS_UNIDIC = True
except ImportError:
    _HAS_UNIDIC = False

# Stanza

try:
    from lexprep.ja.pos_stanza import JaPOSResult, tag_pretokenized_with_stanza
    _HAS_STANZA = True
except ImportError:
    _HAS_STANZA = False


class TestUniDic:
    pytestmark = pytest.mark.skipif(not _HAS_UNIDIC, reason="fugashi/unidic-lite not installed")

    def test_basic_noun(self):
        results = tag_with_unidic(["本"])
        assert len(results) == 1
        r = results[0]
        assert isinstance(r, UniDicResult)
        assert r.text == "本"
        assert r.pos1  # non-empty
        assert r.pos_en  # English mapping
        assert r.error is None

    def test_verb(self):
        results = tag_with_unidic(["食べる"])
        assert len(results) == 1
        assert results[0].pos1  # should be 動詞

    def test_multiple(self):
        results = tag_with_unidic(["猫", "走る", "美しい"])
        assert len(results) == 3
        for r in results:
            assert r.pos1
            assert r.pos_en

    def test_empty_words_skipped(self):
        results = tag_with_unidic(["", "  ", "本"])
        assert len(results) == 1

    def test_lemma_present(self):
        results = tag_with_unidic(["食べる"])
        assert results[0].lemma


class TestStanza:
    pytestmark = pytest.mark.skipif(not _HAS_STANZA, reason="stanza not installed")

    def test_basic(self):
        results = tag_pretokenized_with_stanza(["本"], download_if_missing=True)
        assert len(results) == 1
        r = results[0]
        assert isinstance(r, JaPOSResult)
        assert r.text == "本"
        assert r.upos  # universal POS tag
        assert r.error is None

    def test_multiple(self):
        results = tag_pretokenized_with_stanza(["猫", "食べる"], download_if_missing=True)
        assert len(results) == 2
        for r in results:
            assert r.upos

    def test_empty_skipped(self):
        results = tag_pretokenized_with_stanza(["", "本"], download_if_missing=True)
        assert len(results) == 1
