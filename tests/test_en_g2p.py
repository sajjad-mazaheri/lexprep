"""Tests for English G2P (lexprep.en.g2p).

Requires g2p-en
pip install 'lexprep[en]'
"""

import pytest

try:
    from lexprep.en.g2p import G2PResult, transcribe_words
    _HAS_EN = True
except ImportError:
    _HAS_EN = False

pytestmark = pytest.mark.skipif(not _HAS_EN, reason="English deps not installed (pip install 'lexprep[en]')")


class TestTranscribeWords:
    def test_basic_word(self):
        results = transcribe_words(["hello"])
        assert len(results) == 1
        r = results[0]
        assert isinstance(r, G2PResult)
        assert r.word == "hello"
        assert len(r.pronunciation) > 0
        assert r.error is None

    def test_multiple_words(self):
        results = transcribe_words(["cat", "dog", "fish"])
        assert len(results) == 3
        for r in results:
            assert r.pronunciation  # non-empty

    def test_empty_words_skipped(self):
        results = transcribe_words(["", "  ", "hello"])
        assert len(results) == 1
        assert results[0].word == "hello"

    def test_pronunciation_contains_phonemes(self):
        results = transcribe_words(["hello"])
        pron = results[0].pronunciation
        # ARPAbet phonemes are space-separated
        assert " " in pron or len(pron) > 0
