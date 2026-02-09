"""Tests for Persian POS tagging (lexprep.fa.pos).

Requires: pip install 'lexprep[fa]' (Stanza + Persian model)
"""

import pytest

try:
    from lexprep.fa.pos import POSTagResult, tag_words, tag_words_batch
    _HAS_FA_POS = True
except ImportError:
    _HAS_FA_POS = False

pytestmark = pytest.mark.skipif(not _HAS_FA_POS, reason="Persian POS deps not installed (pip install 'lexprep[fa]')")


class TestTagWords:
    def test_basic(self):
        results = tag_words(["سلام"])
        assert len(results) == 1
        r = results[0]
        assert isinstance(r, POSTagResult)
        assert r.word == "سلام"
        assert r.pos_tag is not None
        assert r.normalized
        assert r.length > 0
        assert r.error is None

    def test_multiple_words(self):
        results = tag_words(["کتاب", "خواندن", "زیبا"])
        assert len(results) == 3
        for r in results:
            assert r.pos_tag is not None

    def test_empty_words_skipped(self):
        results = tag_words(["", "  ", "سلام"])
        assert len(results) == 1

    def test_lemma_present(self):
        results = tag_words(["کتاب"])
        assert results[0].lemma is not None


class TestTagWordsBatch:
    def test_batch(self):
        words = ["سلام", "کتاب", "مادر"]
        results = tag_words_batch(words)
        assert len(results) == 3
        for r in results:
            assert r.pos_tag is not None

    def test_batch_skips_empty(self):
        words = ["", "سلام", "", "کتاب"]
        results = tag_words_batch(words)
        assert len(results) == 2
