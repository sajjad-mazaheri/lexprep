"""Tests for Persian G2P (lexprep.fa.g2p + g2p_cached).

Requires: pip install 'lexprep[fa]'
"""

import pytest

try:
    from lexprep.fa.g2p import G2PResult, transcribe_words
    from lexprep.fa.g2p_cached import CachedPersianG2P, batch_transcribe
    _HAS_FA = True
except ImportError:
    _HAS_FA = False

pytestmark = pytest.mark.skipif(not _HAS_FA, reason="Persian deps not installed (pip install 'lexprep[fa]')")


class TestTranscribeWords:
    def test_basic_word(self):
        results = transcribe_words(["سلام"])
        assert len(results) == 1
        r = results[0]
        assert isinstance(r, G2PResult)
        assert r.word == "سلام"
        assert len(r.pronunciation) > 0
        assert r.error is None

    def test_multiple_words(self):
        results = transcribe_words(["کتاب", "مادر", "پدر"])
        assert len(results) == 3
        for r in results:
            assert r.pronunciation

    def test_empty_words_skipped(self):
        results = transcribe_words(["", "  ", "سلام"])
        assert len(results) == 1
        assert results[0].word == "سلام"


class TestCachedPersianG2P:
    def test_cached_transliterate(self):
        converter = CachedPersianG2P(use_large=True)
        pron = converter.transliterate("سلام", tidy=True)
        assert isinstance(pron, str)
        assert len(pron) > 0

    def test_batch_transcribe(self):
        results = batch_transcribe(["سلام", "کتاب"])
        assert len(results) == 2
        for r in results:
            assert r.pronunciation
            assert r.error is None

    def test_batch_skips_empty(self):
        results = batch_transcribe(["", "  ", "سلام"])
        assert len(results) == 1
