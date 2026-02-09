"""Tests for English POS tagging (lexprep.en.pos).

Requires spaCy + model: pip install 'lexprep[en]' && python -m spacy download en_core_web_sm
"""

import pytest

try:
    import spacy
    spacy.load("en_core_web_sm")
    from lexprep.en.pos import POSResult, tag_words
    _HAS_EN_POS = True
except (ImportError, OSError):
    _HAS_EN_POS = False

pytestmark = pytest.mark.skipif(not _HAS_EN_POS, reason="spaCy + en_core_web_sm not available")


class TestTagWords:
    def test_basic(self):
        results = tag_words(["running"])
        assert len(results) == 1
        r = results[0]
        assert isinstance(r, POSResult)
        assert r.word == "running"
        assert r.pos  # non-empty
        assert r.tag  # non-empty
        assert r.lemma  # non-empty
        assert r.error is None

    def test_noun(self):
        results = tag_words(["dog"])
        assert results[0].pos == "NOUN"

    def test_empty_words_skipped(self):
        results = tag_words(["", "  ", "cat"])
        assert len(results) == 1
        assert results[0].word == "cat"

    def test_multiple_words(self):
        results = tag_words(["run", "beautiful", "quickly"])
        assert len(results) == 3
