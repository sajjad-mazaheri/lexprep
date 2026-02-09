""" tests for the Flask web API (web/app.py).
"""

import io
import json
import os
import sys
import time

import pandas as pd
import pytest

# Add web/ to path so we can import the Flask app
WEB_DIR = os.path.join(os.path.dirname(__file__), "..", "web")
sys.path.insert(0, WEB_DIR)

# Prevent auto-warmup during tests
os.environ.pop("LEXPREP_WARMUP", None)

# Pre-import nltk to avoid circular import issues when PersianG2p loads it
# concurrently from a warmup thread (known nltk bug on Windows).
try:
    import nltk  # noqa: F401
except ImportError:
    pass

from app import app as flask_app  # noqa: E402


# dependency checks


def _has_module(lang):
    """Check if a language module is importable."""
    try:
        if lang == "en":
            from lexprep.en.g2p import transcribe_words  # noqa: F401
            return True
        elif lang == "fa":
            from lexprep.fa.g2p import transcribe_words  # noqa: F401
            return True
        elif lang == "fa_pos":
            from lexprep.fa.pos import tag_words  # noqa: F401
            return True
        elif lang == "ja_unidic":
            from lexprep.ja.pos_unidic import tag_with_unidic  # noqa: F401
            return True
        elif lang == "ja_stanza":
            from lexprep.ja.pos_stanza import tag_pretokenized_with_stanza  # noqa: F401
            return True
    except ImportError:
        return False
    return False


def _has_spacy_model():
    try:
        import spacy
        spacy.load("en_core_web_sm")
        return True
    except (ImportError, OSError):
        return False


# fixtures


@pytest.fixture
def client():
    """Flask test client."""
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


@pytest.fixture
def sample_csv():
    """In-memory CSV file bytes for upload tests."""
    df = pd.DataFrame({"word": ["hello", "world", "test"], "score": [10, 20, 30]})
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


@pytest.fixture
def sample_xlsx():
    """In-memory Excel file bytes for upload tests."""
    df = pd.DataFrame({"word": ["apple", "banana"], "freq": [100, 200]})
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)
    return buf


@pytest.fixture
def persian_csv():
    """CSV with Persian words."""
    df = pd.DataFrame({"word": ["سلام", "کتاب", "مادر"]})
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8")
    buf.seek(0)
    return buf


@pytest.fixture
def japanese_csv():
    """CSV with Japanese words."""
    df = pd.DataFrame({"word": ["本", "食べる", "猫"]})
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8")
    buf.seek(0)
    return buf


# Page routes


class TestPageRoutes:
    """All page routes should return 200."""

    @pytest.mark.parametrize("path", [
        "/", "/about", "/author", "/contribute", "/references",
        "/accuracy", "/sampling",
    ])
    def test_pages_return_200(self, client, path):
        resp = client.get(path)
        assert resp.status_code == 200


# API status and tools


class TestAPIStatus:
    def test_status_endpoint(self, client):
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["api"] == "ok"
        assert "version" in data
        assert "modules" in data
        assert "warmup" in data

    def test_tools_endpoint(self, client):
        resp = client.get("/api/tools")
        assert resp.status_code == 200
        data = resp.get_json()
        # All languages must be present
        assert "fa" in data
        assert "en" in data
        assert "ja" in data
        # Persian tools
        fa_tools = data["fa"]["tools"]
        assert "g2p" in fa_tools
        assert "syllables" in fa_tools
        assert "syllables_phonetic" in fa_tools
        assert "pos" in fa_tools
        # English tools
        en_tools = data["en"]["tools"]
        assert "g2p" in en_tools
        assert "syllables" in en_tools
        assert "pos" in en_tools
        # Japanese tools
        ja_tools = data["ja"]["tools"]
        assert "pos_unidic" in ja_tools
        assert "pos_stanza" in ja_tools

    def test_warmup_status(self, client):
        resp = client.get("/api/warmup/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "started" in data
        assert "completed" in data

    def test_warmup_trigger(self, client):
        resp = client.post("/api/warmup")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] in ("started", "already_started")

    def test_test_persian_endpoint(self, client):
        resp = client.get("/api/test-persian")
        # If Persian deps installed, 200. If not, 500.
        assert resp.status_code in (200, 500)
        if resp.status_code == 200:
            data = resp.get_json()
            assert data["status"] == "ok"
            assert len(data["results"]) == 2


# /api/process validation


class TestProcessValidation:
    def test_missing_words(self, client):
        resp = client.post("/api/process", json={"words": "", "language": "en", "tool": "syllables"})
        assert resp.status_code == 400

    def test_missing_language(self, client):
        resp = client.post("/api/process", json={"words": "hello", "language": "", "tool": "syllables"})
        assert resp.status_code == 400

    def test_missing_tool(self, client):
        resp = client.post("/api/process", json={"words": "hello", "language": "en", "tool": ""})
        assert resp.status_code == 400

    def test_comma_separated_words(self, client):
        """Words can be comma-separated."""
        if not _has_module("en"):
            pytest.skip("English deps not installed")
        resp = client.post("/api/process", json={
            "words": "cat,dog,fish",
            "language": "en",
            "tool": "syllables",
        })
        assert resp.status_code == 200
        assert resp.get_json()["count"] == 3

    def test_newline_separated_words(self, client):
        """Words can be newline-separated."""
        if not _has_module("en"):
            pytest.skip("English deps not installed")
        resp = client.post("/api/process", json={
            "words": "cat\ndog\nfish",
            "language": "en",
            "tool": "syllables",
        })
        assert resp.status_code == 200
        assert resp.get_json()["count"] == 3


# /api/process - English tools


class TestProcessEnglish:
    @pytest.mark.skipif(not _has_module("en"), reason="English deps not installed")
    def test_en_syllables(self, client):
        resp = client.post("/api/process", json={
            "words": "hello\nworld",
            "language": "en",
            "tool": "syllables",
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] == 2
        assert "syllable_count" in data["results"][0]
        assert data["results"][0]["word"] == "hello"
        # "hello" = 2 syllables
        assert data["results"][0]["syllable_count"] == 2

    @pytest.mark.skipif(not _has_module("en"), reason="English deps not installed")
    def test_en_g2p(self, client):
        resp = client.post("/api/process", json={
            "words": "cat",
            "language": "en",
            "tool": "g2p",
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] == 1
        assert "pronunciation" in data["results"][0]
        assert len(data["results"][0]["pronunciation"]) > 0

    @pytest.mark.skipif(not _has_spacy_model(), reason="spaCy model not available")
    def test_en_pos(self, client):
        resp = client.post("/api/process", json={
            "words": "running",
            "language": "en",
            "tool": "pos",
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] == 1
        r = data["results"][0]
        assert "pos" in r
        assert "tag" in r
        assert "lemma" in r
        assert r["word"] == "running"

    @pytest.mark.skipif(not _has_spacy_model(), reason="spaCy model not available")
    def test_en_pos_multiple(self, client):
        resp = client.post("/api/process", json={
            "words": "cat\nrun\nbeautiful",
            "language": "en",
            "tool": "pos",
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] == 3
        for r in data["results"]:
            assert "pos" in r
            assert "tag" in r
            assert "lemma" in r


# /api/process - Persian tools


class TestProcessPersian:
    @pytest.mark.skipif(not _has_module("fa"), reason="Persian deps not installed")
    def test_fa_g2p(self, client):
        resp = client.post("/api/process", json={
            "words": "سلام",
            "language": "fa",
            "tool": "g2p",
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] == 1
        r = data["results"][0]
        assert r["word"] == "سلام"
        assert "pronunciation" in r
        assert len(r["pronunciation"]) > 0

    @pytest.mark.skipif(not _has_module("fa"), reason="Persian deps not installed")
    def test_fa_g2p_multiple(self, client):
        resp = client.post("/api/process", json={
            "words": "کتاب\nمادر\nپدر",
            "language": "fa",
            "tool": "g2p",
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] == 3
        for r in data["results"]:
            assert len(r["pronunciation"]) > 0

    def test_fa_syllables_orthographic(self, client):
        resp = client.post("/api/process", json={
            "words": "کتاب",
            "language": "fa",
            "tool": "syllables",
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] == 1
        r = data["results"][0]
        assert "syllabified" in r
        assert "syllable_count" in r
        assert r["syllable_count"] >= 1

    def test_fa_syllables_orthographic_multiple(self, client):
        resp = client.post("/api/process", json={
            "words": "کتاب\nمادر\nسلام",
            "language": "fa",
            "tool": "syllables",
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] == 3
        for r in data["results"]:
            assert "syllabified" in r
            assert "syllable_count" in r

    @pytest.mark.skipif(not _has_module("fa"), reason="Persian deps not installed")
    def test_fa_syllables_phonetic(self, client):
        resp = client.post("/api/process", json={
            "words": "سلام",
            "language": "fa",
            "tool": "syllables_phonetic",
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] == 1
        r = data["results"][0]
        assert "pronunciation" in r
        assert "syllable_count" in r
        assert r["syllable_count"] >= 1

    @pytest.mark.skipif(not _has_module("fa"), reason="Persian deps not installed")
    def test_fa_syllables_phonetic_multiple(self, client):
        resp = client.post("/api/process", json={
            "words": "کتاب\nمادر",
            "language": "fa",
            "tool": "syllables_phonetic",
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] == 2
        for r in data["results"]:
            assert "pronunciation" in r
            assert "syllable_count" in r

    @pytest.mark.skipif(not _has_module("fa_pos"), reason="Persian POS deps not installed")
    def test_fa_pos(self, client):
        resp = client.post("/api/process", json={
            "words": "کتاب",
            "language": "fa",
            "tool": "pos",
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] == 1
        r = data["results"][0]
        assert r["word"] == "کتاب"
        assert "pos" in r
        assert "normalized" in r
        assert "lemma" in r

    @pytest.mark.skipif(not _has_module("fa_pos"), reason="Persian POS deps not installed")
    def test_fa_pos_multiple(self, client):
        resp = client.post("/api/process", json={
            "words": "کتاب\nخواندن\nزیبا",
            "language": "fa",
            "tool": "pos",
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] == 3
        for r in data["results"]:
            assert "pos" in r
            assert "lemma" in r


# /api/process - Japanese tools


class TestProcessJapanese:
    @pytest.mark.skipif(not _has_module("ja_unidic"), reason="fugashi/unidic-lite not installed")
    def test_ja_pos_unidic(self, client):
        resp = client.post("/api/process", json={
            "words": "本",
            "language": "ja",
            "tool": "pos_unidic",
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] == 1
        r = data["results"][0]
        assert r["word"] == "本"
        assert "pos" in r
        assert "pos_english" in r
        assert "lemma" in r

    @pytest.mark.skipif(not _has_module("ja_unidic"), reason="fugashi/unidic-lite not installed")
    def test_ja_pos_unidic_multiple(self, client):
        resp = client.post("/api/process", json={
            "words": "猫\n食べる\n美しい",
            "language": "ja",
            "tool": "pos_unidic",
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] == 3
        for r in data["results"]:
            assert "pos" in r
            assert "pos_english" in r
            assert "lemma" in r

    @pytest.mark.skipif(not _has_module("ja_stanza"), reason="stanza not installed")
    def test_ja_pos_stanza(self, client):
        resp = client.post("/api/process", json={
            "words": "本",
            "language": "ja",
            "tool": "pos_stanza",
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] == 1
        r = data["results"][0]
        assert r["word"] == "本"
        assert "pos" in r  # upos

    @pytest.mark.skipif(not _has_module("ja_stanza"), reason="stanza not installed")
    def test_ja_pos_stanza_multiple(self, client):
        resp = client.post("/api/process", json={
            "words": "猫\n食べる",
            "language": "ja",
            "tool": "pos_stanza",
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] == 2
        for r in data["results"]:
            assert "pos" in r


# /api/parse-columns


class TestParseColumns:
    def test_csv_columns(self, client, sample_csv):
        resp = client.post("/api/parse-columns", data={
            "file": (sample_csv, "data.csv"),
        }, content_type="multipart/form-data")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "columns" in data
        assert "word" in data["columns"]
        assert data["suggested"] == "word"

    def test_xlsx_columns(self, client, sample_xlsx):
        resp = client.post("/api/parse-columns", data={
            "file": (sample_xlsx, "data.xlsx"),
        }, content_type="multipart/form-data")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "word" in data["columns"]

    def test_no_file_returns_400(self, client):
        resp = client.post("/api/parse-columns", content_type="multipart/form-data")
        assert resp.status_code == 400

    def test_txt_returns_word_column(self, client):
        txt = io.BytesIO(b"apple\nbanana\n")
        resp = client.post("/api/parse-columns", data={
            "file": (txt, "words.txt"),
        }, content_type="multipart/form-data")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["suggested"] == "word"


# /api/download


class TestDownload:
    def test_csv_download(self, client):
        results = [{"word": "hello", "pronunciation": "HH AH L OW"}]
        resp = client.post("/api/download", json={"results": results, "format": "csv"})
        assert resp.status_code == 200
        assert "text/csv" in resp.content_type

    def test_xlsx_download(self, client):
        results = [{"word": "hello", "pronunciation": "HH AH L OW"}]
        resp = client.post("/api/download", json={"results": results, "format": "xlsx"})
        assert resp.status_code == 200

    def test_tsv_download(self, client):
        results = [{"word": "hello"}]
        resp = client.post("/api/download", json={"results": results, "format": "tsv"})
        assert resp.status_code == 200
        assert "tab-separated" in resp.content_type

    def test_empty_results_returns_400(self, client):
        resp = client.post("/api/download", json={"results": [], "format": "csv"})
        assert resp.status_code == 400

    def test_download_multiple_columns(self, client):
        results = [
            {"word": "cat", "pos": "NOUN", "tag": "NN", "lemma": "cat"},
            {"word": "run", "pos": "VERB", "tag": "VB", "lemma": "run"},
        ]
        resp = client.post("/api/download", json={"results": results, "format": "csv"})
        assert resp.status_code == 200
        content = resp.data.decode("utf-8")
        assert "pos" in content
        assert "lemma" in content


# /api/process-file (sync)


class TestProcessFile:
    def test_no_file_returns_400(self, client):
        resp = client.post("/api/process-file", data={
            "language": "en", "tool": "syllables",
        }, content_type="multipart/form-data")
        assert resp.status_code == 400

    def test_missing_language_returns_400(self, client, sample_csv):
        resp = client.post("/api/process-file", data={
            "file": (sample_csv, "data.csv"),
            "language": "",
            "tool": "syllables",
            "word_column": "word",
        }, content_type="multipart/form-data")
        assert resp.status_code == 400

    def test_bad_column_returns_400(self, client, sample_csv):
        if not _has_module("en"):
            pytest.skip("English deps not installed")
        resp = client.post("/api/process-file", data={
            "file": (sample_csv, "data.csv"),
            "language": "en",
            "tool": "syllables",
            "word_column": "nonexistent",
        }, content_type="multipart/form-data")
        assert resp.status_code == 400

    def test_unsupported_format(self, client):
        fake = io.BytesIO(b"some data")
        resp = client.post("/api/process-file", data={
            "file": (fake, "data.json"),
            "language": "en",
            "tool": "syllables",
            "word_column": "word",
        }, content_type="multipart/form-data")
        assert resp.status_code == 400

    @pytest.mark.skipif(not _has_module("en"), reason="English deps not installed")
    def test_en_syllables_file_csv(self, client, sample_csv):
        resp = client.post("/api/process-file", data={
            "file": (sample_csv, "data.csv"),
            "language": "en",
            "tool": "syllables",
            "word_column": "word",
        }, content_type="multipart/form-data")
        assert resp.status_code == 200
        assert "text/csv" in resp.content_type or "spreadsheet" in resp.content_type

    @pytest.mark.skipif(not _has_module("en"), reason="English deps not installed")
    def test_en_syllables_file_txt(self, client):
        txt = io.BytesIO(b"hello\nworld\ntest\n")
        resp = client.post("/api/process-file", data={
            "file": (txt, "words.txt"),
            "language": "en",
            "tool": "syllables",
            "word_column": "word",
        }, content_type="multipart/form-data")
        assert resp.status_code == 200

    @pytest.mark.skipif(not _has_module("en"), reason="English deps not installed")
    def test_en_g2p_file(self, client, sample_csv):
        resp = client.post("/api/process-file", data={
            "file": (sample_csv, "data.csv"),
            "language": "en",
            "tool": "g2p",
            "word_column": "word",
        }, content_type="multipart/form-data")
        assert resp.status_code == 200

    @pytest.mark.skipif(not _has_spacy_model(), reason="spaCy model not available")
    def test_en_pos_file(self, client, sample_csv):
        resp = client.post("/api/process-file", data={
            "file": (sample_csv, "data.csv"),
            "language": "en",
            "tool": "pos",
            "word_column": "word",
        }, content_type="multipart/form-data")
        assert resp.status_code == 200

    @pytest.mark.skipif(not _has_module("fa"), reason="Persian deps not installed")
    def test_fa_g2p_file(self, client, persian_csv):
        resp = client.post("/api/process-file", data={
            "file": (persian_csv, "fa.csv"),
            "language": "fa",
            "tool": "g2p",
            "word_column": "word",
        }, content_type="multipart/form-data")
        assert resp.status_code == 200

    def test_fa_syllables_file(self, client, persian_csv):
        resp = client.post("/api/process-file", data={
            "file": (persian_csv, "fa.csv"),
            "language": "fa",
            "tool": "syllables",
            "word_column": "word",
        }, content_type="multipart/form-data")
        assert resp.status_code == 200

    @pytest.mark.skipif(not _has_module("fa"), reason="Persian deps not installed")
    def test_fa_syllables_phonetic_file(self, client, persian_csv):
        resp = client.post("/api/process-file", data={
            "file": (persian_csv, "fa.csv"),
            "language": "fa",
            "tool": "syllables_phonetic",
            "word_column": "word",
        }, content_type="multipart/form-data")
        assert resp.status_code == 200

    @pytest.mark.skipif(not _has_module("fa_pos"), reason="Persian POS deps not installed")
    def test_fa_pos_file(self, client, persian_csv):
        resp = client.post("/api/process-file", data={
            "file": (persian_csv, "fa.csv"),
            "language": "fa",
            "tool": "pos",
            "word_column": "word",
        }, content_type="multipart/form-data")
        assert resp.status_code == 200

    @pytest.mark.skipif(not _has_module("ja_unidic"), reason="fugashi/unidic-lite not installed")
    def test_ja_pos_unidic_file(self, client, japanese_csv):
        resp = client.post("/api/process-file", data={
            "file": (japanese_csv, "ja.csv"),
            "language": "ja",
            "tool": "pos_unidic",
            "word_column": "word",
        }, content_type="multipart/form-data")
        assert resp.status_code == 200

    @pytest.mark.skipif(not _has_module("ja_stanza"), reason="stanza not installed")
    def test_ja_pos_stanza_file(self, client, japanese_csv):
        resp = client.post("/api/process-file", data={
            "file": (japanese_csv, "ja.csv"),
            "language": "ja",
            "tool": "pos_stanza",
            "word_column": "word",
        }, content_type="multipart/form-data")
        assert resp.status_code == 200

    @pytest.mark.skipif(not _has_module("en"), reason="English deps not installed")
    def test_xlsx_upload(self, client, sample_xlsx):
        resp = client.post("/api/process-file", data={
            "file": (sample_xlsx, "data.xlsx"),
            "language": "en",
            "tool": "syllables",
            "word_column": "word",
        }, content_type="multipart/form-data")
        assert resp.status_code == 200


# /api/process-file-async


class TestAsyncProcessing:
    def test_no_file_returns_400(self, client):
        resp = client.post("/api/process-file-async", data={
            "language": "en", "tool": "syllables",
        }, content_type="multipart/form-data")
        assert resp.status_code == 400

    def test_missing_language_returns_400(self, client, sample_csv):
        resp = client.post("/api/process-file-async", data={
            "file": (sample_csv, "data.csv"),
            "language": "",
            "tool": "syllables",
        }, content_type="multipart/form-data")
        assert resp.status_code == 400

    @pytest.mark.skipif(not _has_module("en"), reason="English deps not installed")
    def test_async_creates_job(self, client, sample_csv):
        resp = client.post("/api/process-file-async", data={
            "file": (sample_csv, "data.csv"),
            "language": "en",
            "tool": "syllables",
            "word_column": "word",
        }, content_type="multipart/form-data")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "job_id" in data
        assert data["status"] == "started"

    @pytest.mark.skipif(not _has_module("en"), reason="English deps not installed")
    def test_async_job_status(self, client, sample_csv):
        resp = client.post("/api/process-file-async", data={
            "file": (sample_csv, "data.csv"),
            "language": "en",
            "tool": "syllables",
            "word_column": "word",
        }, content_type="multipart/form-data")
        job_id = resp.get_json()["job_id"]
        time.sleep(1)
        resp2 = client.get(f"/api/job/{job_id}")
        assert resp2.status_code == 200
        data = resp2.get_json()
        assert data["status"] in ("running", "completed", "error")

    def test_nonexistent_job_returns_404(self, client):
        resp = client.get("/api/job/nonexistent-id")
        assert resp.status_code == 404

    def test_download_nonexistent_job_returns_404(self, client):
        resp = client.get("/api/job/nonexistent-id/download")
        assert resp.status_code == 404


# /api/track


class TestTracking:
    def test_track_event(self, client):
        resp = client.post("/api/track", json={"page": "test_page"})
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "ok"


# /api/sampling/*


class TestSamplingAPI:
    def test_parse_file_csv(self, client, sample_csv):
        resp = client.post("/api/sampling/parse-file", data={
            "file": (sample_csv, "data.csv"),
        }, content_type="multipart/form-data")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "columns" in data
        assert "row_count" in data
        assert data["row_count"] == 3

    def test_parse_file_no_file(self, client):
        resp = client.post("/api/sampling/parse-file", content_type="multipart/form-data")
        assert resp.status_code == 400

    def test_stratified_quantile(self, client):
        df = pd.DataFrame({"word": [f"w{i}" for i in range(50)], "score": list(range(50))})
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)

        resp = client.post("/api/sampling/stratified", data={
            "file": (buf, "data.csv"),
            "score_col": "score",
            "n_total": "15",
            "mode": "quantile",
            "bins": "3",
            "allocation": "equal",
            "random_state": "42",
        }, content_type="multipart/form-data")
        assert resp.status_code == 200
        assert "spreadsheet" in resp.content_type or "excel" in resp.content_type.lower()

    def test_stratified_proportional_allocation(self, client):
        df = pd.DataFrame({"word": [f"w{i}" for i in range(60)], "score": list(range(60))})
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)

        resp = client.post("/api/sampling/stratified", data={
            "file": (buf, "data.csv"),
            "score_col": "score",
            "n_total": "18",
            "mode": "quantile",
            "bins": "3",
            "allocation": "proportional",
            "random_state": "42",
        }, content_type="multipart/form-data")
        assert resp.status_code == 200

    def test_stratified_missing_score_col(self, client, sample_csv):
        resp = client.post("/api/sampling/stratified", data={
            "file": (sample_csv, "data.csv"),
            "score_col": "",
            "n_total": "5",
        }, content_type="multipart/form-data")
        assert resp.status_code == 400

    def test_stratified_custom_ranges(self, client):
        df = pd.DataFrame({"word": [f"w{i}" for i in range(50)], "score": list(range(50))})
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)

        ranges = json.dumps([
            {"lower": 0, "upper": 20, "lower_inclusive": True, "upper_inclusive": True},
            {"lower": 21, "upper": 49, "lower_inclusive": True, "upper_inclusive": True},
        ])

        resp = client.post("/api/sampling/stratified", data={
            "file": (buf, "data.csv"),
            "score_col": "score",
            "n_total": "10",
            "mode": "custom",
            "ranges": ranges,
            "allocation": "equal",
            "random_state": "42",
        }, content_type="multipart/form-data")
        assert resp.status_code == 200

    def test_shuffle_needs_two_files(self, client, sample_csv):
        resp = client.post("/api/sampling/shuffle", data={
            "files": (sample_csv, "one.csv"),
            "seed": "42",
        }, content_type="multipart/form-data")
        assert resp.status_code == 400

    def test_shuffle_two_files(self, client):
        df1 = pd.DataFrame({"word": ["a", "b", "c"], "x": [1, 2, 3]})
        df2 = pd.DataFrame({"word": ["d", "e", "f"], "x": [4, 5, 6]})

        buf1 = io.BytesIO()
        df1.to_csv(buf1, index=False)
        buf1.seek(0)

        buf2 = io.BytesIO()
        df2.to_csv(buf2, index=False)
        buf2.seek(0)

        resp = client.post("/api/sampling/shuffle", data={
            "files": [(buf1, "f1.csv"), (buf2, "f2.csv")],
            "seed": "42",
        }, content_type="multipart/form-data")
        assert resp.status_code == 200
        assert "zip" in resp.content_type


# Admin


class TestAdmin:
    def test_admin_disabled_without_secret(self, client):
        resp = client.get("/api/admin/stats")
        assert resp.status_code in (401, 403)

    def test_admin_page_without_secret(self, client):
        resp = client.get("/admin")
        assert resp.status_code in (200, 403)

    def test_admin_bad_token(self, client):
        resp = client.get("/api/admin/stats", headers={
            "Authorization": "Bearer wrong_token"
        })
        assert resp.status_code in (401, 403)
