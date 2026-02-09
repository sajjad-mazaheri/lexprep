"""Tests for Japanese POS mapping (lexprep.ja.pos_map).

"""

from lexprep.ja.pos_map import (
    DEFAULT_POS_MAP,
    map_pos_to_english,
    normalize_pos_japanese,
)


class TestNormalizePosJapanese:
    def test_strips_whitespace(self):
        assert normalize_pos_japanese("  名詞  ") == "名詞"

    def test_removes_fullwidth_spaces(self):
        assert normalize_pos_japanese("名詞\u3000") == "名詞"

    def test_none_returns_empty(self):
        assert normalize_pos_japanese(None) == ""


class TestMapPosToEnglish:
    def test_known_tags(self):
        assert map_pos_to_english("名詞") == "noun"
        assert map_pos_to_english("動詞") == "verb"
        assert map_pos_to_english("形容詞") == "adjective"
        assert map_pos_to_english("副詞") == "adverb"
        assert map_pos_to_english("助詞") == "particle"

    def test_unknown_tag(self):
        assert map_pos_to_english("unknown_tag") == "Unknown"

    def test_custom_mapping(self):
        custom = {"テスト": "test"}
        assert map_pos_to_english("テスト", mapping=custom) == "test"

    def test_all_defaults_covered(self):
        for jp, en in DEFAULT_POS_MAP.items():
            assert map_pos_to_english(jp) == en
