import hashlib
import hmac
import io
import json
import logging
import os
import re
import secrets
import sqlite3
import threading
import time as _time
import uuid
from datetime import datetime
from functools import wraps

import pandas as pd
from flask import Flask, Response, jsonify, render_template, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from markupsafe import Markup
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename

from lexprep.length import LENGTH_METHOD, compute_length_chars, length_distribution
from lexprep.manifest import (
    TOOL_REGISTRY,
    build_manifest,
    get_libraries_for_tool,
    registry_key,
    sanitize_basename,
    utc_now,
)
from lexprep.packaging import _df_to_bytes, build_zip, make_zip_filename

logger = logging.getLogger(__name__)

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['JSON_AS_ASCII'] = False
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', secrets.token_hex(32))
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB upload limit

# Trust one level of proxy (nginx/reverse proxy)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)

# CORS: restrict to production domain + localhost for development
_cors_origins = os.environ.get(
    'CORS_ORIGINS',
    'https://lexprep.net,http://localhost:5000,http://127.0.0.1:5000',
).split(',')
CORS(app, origins=[o.strip() for o in _cors_origins if o.strip()])

# Rate limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=['200 per minute'],
    storage_uri='memory://',
    enabled=os.environ.get('LEXPREP_RATE_LIMIT', 'true').lower() != 'false',
)

# Version (single source of truth: src/lexprep/__init__.py)
try:
    from lexprep import __version__
    VERSION = __version__
except ImportError:
    VERSION = '1.0.0'

ADMIN_SECRET = os.environ.get('LEXPREP_ADMIN_SECRET')
ADMIN_ENABLED = ADMIN_SECRET is not None and len(ADMIN_SECRET) >= 32

# Google Analytics tracking ID (set to empty string to disable)
GA_TRACKING_ID = os.environ.get('GA_TRACKING_ID', 'G-KH8H8BJF9C')

# Analytics database path
DB_PATH = os.environ.get('LEXPREP_DB_PATH',
                         os.path.join(os.path.dirname(__file__), 'analytics.db'))

# Input validation allowlists
VALID_LANGUAGES = {'en', 'fa', 'ja'}
VALID_TOOLS = {
    'g2p', 'syllables', 'syllables_phonetic', 'pos',
    'pos_unidic', 'pos_stanza', 'length',
}
ALLOWED_EXTENSIONS = {'csv', 'tsv', 'txt', 'xlsx', 'xls'}

_LANG_LOOKUP: dict[str, str] = {lang: lang for lang in VALID_LANGUAGES}
_TOOL_LOOKUP: dict[str, str] = {t: t for t in VALID_TOOLS}
_FMT_LOOKUP: dict[str, str] = {'csv': 'csv', 'tsv': 'tsv', 'xlsx': 'xlsx'}

# Sanitise user-supplied column names before storing them in manifests
# or using them in filenames. Only keep word characters, spaces,
# dots and hyphens; replace everything else with "_".
_COL_NAME_UNSAFE = re.compile(r'[^\w\s.\-]')

# Privacy: salted IP hashing (generate once per server lifetime)
_IP_SALT = os.environ.get('LEXPREP_IP_SALT', secrets.token_hex(16))

# Processing limits
MAX_WORDS = 5000


# Model Preloading and Caching
# Cache for loaded models
_model_cache = {
    'fa_g2p': None,
    'fa_g2p_converter': None,
    'fa_pos': None,
    'en_g2p': None,
    'en_pos': None,
    'ja_unidic': None,
    'ja_stanza': None,
}
_model_lock = threading.Lock()
_warmup_status = {
    'started': False,
    'completed': False,
    'errors': [],
    'loaded': []
}

# Auto-warmup for gunicorn (when LEXPREP_WARMUP=true)
_auto_warmup_done = False


def preload_models():
    """Preload all NLP models at startup to avoid cold start delays."""
    global _warmup_status
    _warmup_status['started'] = True
    _warmup_status['errors'] = []
    _warmup_status['loaded'] = []

    # Persian G2P
    try:
        from lexprep.fa.g2p_cached import CachedPersianG2P
        print("[Warmup] Loading Persian G2P converter...")
        converter = CachedPersianG2P(use_large=True)
        _ = converter.transliterate('سلام', tidy=True)
        _model_cache['fa_g2p_converter'] = converter
        _model_cache['fa_g2p'] = True
        _warmup_status['loaded'].append('fa_g2p')
        print("[Warmup] Persian G2P loaded")
    except Exception as e:
        _warmup_status['errors'].append('fa_g2p: failed to load')
        logger.exception("Warmup failed for fa_g2p: %s", e)

    # Persian POS (Stanza)
    try:
        from lexprep.fa.pos import tag_words as fa_tag_words
        fa_tag_words(['سلام'])
        _model_cache['fa_pos'] = True
        _warmup_status['loaded'].append('fa_pos')
        print("[Warmup] Persian POS loaded (Stanza)")
    except Exception as e:
        _warmup_status['errors'].append('fa_pos: failed to load')
        logger.exception("Warmup failed for fa_pos: %s", e)

    # English G2P
    try:
        from lexprep.en.g2p import transcribe_words
        transcribe_words(['hello'])
        _model_cache['en_g2p'] = True
        _warmup_status['loaded'].append('en_g2p')
        print("[Warmup] English G2P loaded")
    except Exception as e:
        _warmup_status['errors'].append('en_g2p: failed to load')
        logger.exception("Warmup failed for en_g2p: %s", e)

    # English POS (spaCy)
    try:
        from lexprep.en.pos import tag_words
        tag_words(['hello'])
        _model_cache['en_pos'] = True
        _warmup_status['loaded'].append('en_pos')
        print("[Warmup] English POS loaded")
    except Exception as e:
        _warmup_status['errors'].append('en_pos: failed to load')
        logger.exception("Warmup failed for en_pos: %s", e)

    # Japanese UniDic
    try:
        from lexprep.ja.pos_unidic import tag_with_unidic
        tag_with_unidic(['日本語'])
        _model_cache['ja_unidic'] = True
        _warmup_status['loaded'].append('ja_unidic')
        print("[Warmup] Japanese UniDic loaded")
    except Exception as e:
        _warmup_status['errors'].append('ja_unidic: failed to load')
        logger.exception("Warmup failed for ja_unidic: %s", e)

    # Japanese Stanza (optional - can be slow)
    try:
        from lexprep.ja.pos_stanza import tag_pretokenized_with_stanza
        tag_pretokenized_with_stanza(['日本語'], download_if_missing=True)
        _model_cache['ja_stanza'] = True
        _warmup_status['loaded'].append('ja_stanza')
        print("[Warmup] Japanese Stanza loaded")
    except Exception as e:
        _warmup_status['errors'].append('ja_stanza: failed to load')
        logger.exception("Warmup failed for ja_stanza: %s", e)

    _warmup_status['completed'] = True
    print(f"[Warmup] Complete. Loaded: {_warmup_status['loaded']}")


# Auto-warmup at module load (for gunicorn with --preload)
def _do_auto_warmup():
    global _auto_warmup_done
    if _auto_warmup_done:
        return
    if os.environ.get('LEXPREP_WARMUP', '').lower() == 'true':
        print("[Startup] Running model warmup (gunicorn preload)...")
        import sys
        sys.stdout.flush()
        preload_models()
        _auto_warmup_done = True

_do_auto_warmup()


# Background job storage for async processing
_jobs = {}
_jobs_lock = threading.Lock()
_JOB_TTL_SECONDS = 1800  # 30 minutes


def _cleanup_expired_jobs():
    """Remove jobs older than TTL to prevent memory exhaustion."""
    while True:
        _time.sleep(300)  # Run every 5 minutes
        now = datetime.now()
        with _jobs_lock:
            expired = [
                jid for jid, job in _jobs.items()
                if (now - datetime.fromisoformat(job['created_at'])).total_seconds()
                > _JOB_TTL_SECONDS
            ]
            for jid in expired:
                del _jobs[jid]
            if expired:
                logger.info('Cleaned up %d expired jobs', len(expired))


_cleanup_thread = threading.Thread(target=_cleanup_expired_jobs, daemon=True)
_cleanup_thread.start()


def init_db():
    """Initialize SQLite database for analytics"""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS page_views (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            page TEXT,
            ip_hash TEXT,
            user_agent TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')
        conn.execute('''CREATE TABLE IF NOT EXISTS tool_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            language TEXT,
            tool TEXT,
            word_count INTEGER,
            ip_hash TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')


def hash_ip(ip):
    """Hash IP for privacy (salted to prevent brute-force reversal)"""
    return hashlib.sha256((_IP_SALT + ip).encode()).hexdigest()[:16]


# Valid page names for analytics
_VALID_PAGES = {
    'home', 'about', 'author', 'contribute', 'references',
    'accuracy', 'sampling', 'admin', 'unknown',
}


def log_page_view(page):
    """Log a page view"""
    try:
        if page not in _VALID_PAGES:
            page = 'unknown'
        ip = request.remote_addr or 'unknown'
        ua = request.headers.get('User-Agent', '')[:200]
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                'INSERT INTO page_views (page, ip_hash, user_agent) VALUES (?, ?, ?)',
                (page, hash_ip(ip), ua),
            )
    except Exception:
        logger.exception('Failed to log page view for page: %s', page)


def log_tool_usage(language, tool, word_count):
    """Log tool usage"""
    try:
        ip = request.remote_addr or 'unknown'
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                'INSERT INTO tool_usage (language, tool, word_count, ip_hash) VALUES (?, ?, ?, ?)',
                (language, tool, word_count, hash_ip(ip)),
            )
    except Exception:
        logger.exception('Failed to log tool usage: %s/%s', language, tool)


def admin_required(f):
    """Decorator for admin endpoints - requires secure environment variable"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Admin must be explicitly enabled with a secure secret
        if not ADMIN_ENABLED:
            return jsonify({'error': 'Admin functionality is disabled. Set LEXPREP_ADMIN_SECRET environment variable (min 32 chars).'}), 403

        auth = request.headers.get('Authorization', '')
        if not auth.startswith('Bearer '):
            return jsonify({'error': 'Authorization header required'}), 401

        token = auth[7:]  # Remove 'Bearer ' prefix
        # Use constant-time comparison to prevent timing attacks
        if not hmac.compare_digest(token, ADMIN_SECRET):
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated_function


# Initialize database on startup
init_db()


@app.context_processor
def inject_globals():
    return {'ga_tracking_id': GA_TRACKING_ID}


# Page Routes

@app.route('/')
def index():
    log_page_view('home')
    return render_template('index.html', active_page='home')


@app.route('/about')
def about():
    log_page_view('about')
    return render_template('about.html', active_page='about',
        page_title=Markup('About <span class="gradient-text">lexprep</span>'),
        page_subtitle='A toolkit designed for psycholinguistic research and linguistic data preparation')


@app.route('/author')
def author():
    log_page_view('author')
    return render_template('author.html', active_page='author',
        page_title=Markup('Meet the <span class="gradient-text">Author</span>'),
        page_subtitle='The person behind lexprep')


@app.route('/contribute')
def contribute():
    log_page_view('contribute')
    return render_template('contribute.html', active_page='contribute',
        page_title=Markup('<span class="gradient-text">Contribute</span> to lexprep'),
        page_subtitle='Help us build better tools for linguistic research')


@app.route('/references')
def references():
    log_page_view('references')
    return render_template('references.html', active_page='references',
        page_title=Markup('<span class="gradient-text">References</span> &amp; Libraries'),
        page_subtitle='The excellent open-source libraries that power lexprep')


@app.route('/accuracy')
def accuracy():
    log_page_view('accuracy')
    return render_template('accuracy.html', active_page='accuracy',
        page_title=Markup('Tool <span class="gradient-text">Accuracy</span>'),
        page_subtitle='Evaluation metrics and benchmarks for each tool')


@app.route('/sampling')
def sampling():
    log_page_view('sampling')
    return render_template('sampling.html', active_page='sampling',
        page_title=Markup('<span class="gradient-text">Sampling Tools</span>'),
        page_subtitle='Statistical sampling methods for experimental stimulus selection')


@app.route('/admin')
def admin():
    if not ADMIN_ENABLED:
        return render_template('admin_disabled.html'), 403
    return render_template('admin.html', active_page='admin',
        page_title=Markup('<span class="gradient-text">Admin</span> Panel'))


# ============== ZIP/Manifest Helpers ==============


def _safe_ext(raw: str) -> str:
    """Extract a validated extension from a raw upload filename.

    Returns one of ALLOWED_EXTENSIONS or ``''`` when the extension is
    not recognised.  The result is always a hardcoded constant, which
    CodeQL recognises as a ConstCompare sanitizer barrier cutting
    tainted data-flow.
    """
    name = secure_filename(raw) or ''
    ext = name.rsplit('.', 1)[-1].lower() if '.' in name else ''
    if ext in ALLOWED_EXTENSIONS:
        return ext
    return ''


def _manifest_tool_name(language, tool):
    """Return the manifest-friendly tool name from the registry.

    This helper keeps the zip-filename computation independent of the
    (potentially tainted) manifest dict.
    """
    rkey = registry_key(language, tool) if language else tool
    spec = TOOL_REGISTRY.get(rkey)
    return spec.manifest_name if spec else tool


def _safe_col_name(raw: str) -> str:
    """Sanitize a column name by replacing unsafe characters with "_".
"""
    return _COL_NAME_UNSAFE.sub('_', raw) if raw else ''


def _download_bytes_response(data: bytes, *, mimetype: str, download_name: str) -> Response:
    """Return an attachment response for in-memory bytes.

    avoid ``flask.send_file`` for in-memory payloads -
    path-injection analysis does not treat the body bytes as a path argument.
    """
    response = Response(data, mimetype=mimetype)
    response.headers.set('Content-Disposition', 'attachment', filename=download_name)
    return response



def _compute_summary(tool, result_df):
    """Compute tool-dependent summary stats for manifest."""
    if tool in ('pos', 'pos_unidic', 'pos_stanza'):
        if 'pos' in result_df.columns:
            return {'pos_distribution': result_df['pos'].value_counts().to_dict()}
    elif tool in ('syllables', 'syllables_phonetic'):
        if 'syllable_count' in result_df.columns:
            return {'syllable_distribution': result_df['syllable_count'].value_counts().to_dict()}
    elif tool == 'length':
        if 'length_chars' in result_df.columns:
            lengths = result_df['length_chars'].tolist()
            dist = length_distribution(lengths)
            summary = {'length_method': LENGTH_METHOD}
            if dist:
                summary['length_distribution'] = dist.to_dict()
            return summary
    return None


def _get_added_columns(tool, result_columns):
    """Get the list of added columns for a tool (excluding 'word')."""
    return [c for c in result_columns if c != 'word']


def _build_zip_response(
    tool, language, word_column, df, output_df, result_df,
    ext=None, timestamp=None, original_filename='upload',
):
    """Build a ZIP response for language tool endpoints."""
    ts = timestamp or utc_now()
    if not ext:
        ext = 'xlsx'
    safe_name = secure_filename(original_filename) or 'upload'
    safe_base = safe_name.rsplit('.', 1)[0] if '.' in safe_name else safe_name

    # Language-agnostic tools should not record a language
    manifest_language = None if tool == 'length' else language

    # Pre-compute ZIP filename from trusted values (breaks taint chain)
    manifest_tool = _manifest_tool_name(manifest_language, tool)
    zip_fname = make_zip_filename(safe_base, manifest_tool, manifest_language, ts)

    added_cols = _get_added_columns(tool, result_df.columns)
    summary = _compute_summary(tool, result_df)

    manifest = build_manifest(
        tool_key=tool,
        language=manifest_language,
        original_filename=safe_name,
        file_type=ext,
        row_count=len(df),
        column_mapping={'word_column': _safe_col_name(word_column)},
        added_columns=added_cols,
        libraries=get_libraries_for_tool(language, tool),
        timestamp=ts,
        summary=summary,
    )
    zip_bytes, zip_name = build_zip(
        manifest=manifest,
        main_df=output_df,
        input_basename=safe_base,
        output_ext=ext,
        zip_filename=zip_fname,
    )
    return _download_bytes_response(
        zip_bytes,
        mimetype='application/zip',
        download_name=zip_name,
    )


# ============== API Routes ==============

@app.route('/api/test-persian', methods=['GET'])
def test_persian():
    """Quick test endpoint for Persian G2P"""
    try:
        from lexprep.fa.g2p import transcribe_words
        result = transcribe_words(['سلام', 'کتاب'])
        return jsonify({
            'status': 'ok',
            'results': [{'word': r.word, 'pronunciation': r.pronunciation, 'error': r.error} for r in result]
        })
    except Exception:
        logger.exception('Error in test_persian')
        return jsonify({'status': 'error', 'error': 'An internal error occurred'}), 500




@app.route('/api/status', methods=['GET'])
def api_status():
    """Check if API and modules are working"""
    status = {
        'api': 'ok',
        'version': VERSION,
        'modules': {},
        'warmup': {
            'started': _warmup_status['started'],
            'completed': _warmup_status['completed'],
            'loaded': _warmup_status['loaded'],
            'error_count': len(_warmup_status['errors'])
        }
    }

    try:
        from lexprep.fa.g2p import transcribe_words
        status['modules']['fa'] = 'ok'
    except ImportError:
        status['modules']['fa'] = 'not installed'

    try:
        from lexprep.en.g2p import transcribe_words
        status['modules']['en'] = 'ok'
    except ImportError:
        status['modules']['en'] = 'not installed'

    try:
        from lexprep.ja.pos_unidic import tag_with_unidic
        status['modules']['ja'] = 'ok'
    except ImportError:
        status['modules']['ja'] = 'not installed'

    return jsonify(status)


@app.route('/api/warmup', methods=['POST'])
@limiter.limit('5 per hour')
def warmup():
    """Trigger model warmup - call this after server starts"""
    if _warmup_status['started']:
        return jsonify({
            'status': 'already_started',
            'completed': _warmup_status['completed'],
            'loaded': _warmup_status['loaded'],
            'error_count': len(_warmup_status['errors'])
        })

    # Run warmup in background thread
    thread = threading.Thread(target=preload_models, daemon=True)
    thread.start()

    return jsonify({
        'status': 'started',
        'message': 'Model warmup started in background'
    })


@app.route('/api/warmup/status', methods=['GET'])
def warmup_status():
    """Check warmup status"""
    return jsonify({
        'started': _warmup_status['started'],
        'completed': _warmup_status['completed'],
        'loaded': _warmup_status['loaded'],
        'error_count': len(_warmup_status['errors'])
    })


@app.route('/api/tools', methods=['GET'])
def get_tools():
    """Return available tools per language"""
    tools = {
        'fa': {
            'name': 'Persian (فارسی)',
            'flag': '/static/images/iran-flag-icon.svg',
            'tools': {
                'g2p': {'name': 'G2P Transcription', 'description': 'Convert words to phonetic representations'},
                'syllables': {'name': 'Syllable Count', 'description': 'Count syllables using orthographic method'},
                'syllables_phonetic': {'name': 'Syllable Count (Phonetic)', 'description': 'Count syllables with automatic G2P'},
                'pos': {'name': 'POS Tagging', 'description': 'Part-of-speech tagging with Stanza (UD tags)'},
                'length': {'name': 'Word Length', 'description': 'Count Unicode codepoints (length_chars)'},
            }
        },
        'en': {
            'name': 'English',
            'flag': '/static/images/united-states-flag-icon.svg',
            'tools': {
                'g2p': {'name': 'G2P (ARPAbet)', 'description': 'Convert to ARPAbet phonemes'},
                'syllables': {'name': 'Syllable Count', 'description': 'Count syllables using pyphen'},
                'pos': {'name': 'POS Tagging', 'description': 'Part-of-speech tagging with spaCy'},
                'length': {'name': 'Word Length', 'description': 'Count Unicode codepoints (length_chars)'},
            }
        },
        'ja': {
            'name': 'Japanese (日本語)',
            'flag': '/static/images/japan-flag-icon.svg',
            'tools': {
                'pos_unidic': {'name': 'POS (UniDic)', 'description': 'Detailed Japanese POS tags'},
                'pos_stanza': {'name': 'POS (Stanza)', 'description': 'Universal POS tags'},
                'length': {'name': 'Word Length', 'description': 'Count Unicode codepoints (length_chars)'},
            }
        }
    }
    return jsonify(tools)


@app.route('/api/process', methods=['POST'])
@limiter.limit('30 per minute')
def process_words():
    """Process word list with selected tool"""
    try:
        data = request.json
        words_text = data.get('words', '')
        language = data.get('language', '')
        tool = data.get('tool', '')

        # Parse words (one per line or comma-separated)
        words = [w.strip() for w in words_text.replace(',', '\n').split('\n') if w.strip()]

        if not words:
            return jsonify({'error': 'No words provided'}), 400

        if len(words) > MAX_WORDS:
            return jsonify({'error': f'Maximum {MAX_WORDS} words allowed per request'}), 400

        if language not in VALID_LANGUAGES:
            return jsonify({'error': 'Invalid language'}), 400
        if tool not in VALID_TOOLS:
            return jsonify({'error': 'Invalid tool'}), 400

        # Log usage
        log_tool_usage(language, tool, len(words))

        # Process based on language and tool
        results = []

        if language == 'fa':
            if tool == 'g2p':
                from lexprep.fa.g2p import transcribe_words
                transcriptions = transcribe_words(words)
                for i, w in enumerate(words):
                    r = transcriptions[i]
                    results.append({
                        'word': w,
                        'pronunciation': r.pronunciation,
                        'error': r.error or ''
                    })

            elif tool == 'syllables':
                from lexprep.fa.syllables import syllabify_orthographic
                for w in words:
                    syllabified, count = syllabify_orthographic(w)
                    results.append({
                        'word': w,
                        'syllabified': syllabified,
                        'syllable_count': count
                    })

            elif tool == 'syllables_phonetic':
                from lexprep.fa.g2p import transcribe_words
                from lexprep.fa.syllables import count_syllables_from_pronunciation
                transcriptions = transcribe_words(words)
                for i, w in enumerate(words):
                    r = transcriptions[i]
                    count = count_syllables_from_pronunciation(r.pronunciation)
                    results.append({
                        'word': w,
                        'pronunciation': r.pronunciation,
                        'syllable_count': count
                    })

            elif tool == 'pos':
                results = process_persian_pos(words)

        elif language == 'en':
            if tool == 'g2p':
                from lexprep.en.g2p import transcribe_words
                transcriptions = transcribe_words(words)
                for i, w in enumerate(words):
                    r = transcriptions[i]
                    results.append({
                        'word': w,
                        'pronunciation': r.pronunciation,
                        'error': r.error or ''
                    })

            elif tool == 'syllables':
                from lexprep.en.syllables import count_syllables
                for w in words:
                    count = count_syllables(w)
                    results.append({
                        'word': w,
                        'syllable_count': count
                    })

            elif tool == 'pos':
                from lexprep.en.pos import tag_words
                tags = tag_words(words)
                for i, w in enumerate(words):
                    r = tags[i]
                    results.append({
                        'word': w,
                        'pos': r.pos,
                        'tag': r.tag,
                        'lemma': r.lemma
                    })

        elif language == 'ja':
            if tool == 'pos_unidic':
                from lexprep.ja.pos_map import map_pos_to_english
                from lexprep.ja.pos_unidic import tag_with_unidic
                tags = tag_with_unidic(words)
                for i, w in enumerate(words):
                    r = tags[i]
                    results.append({
                        'word': w,
                        'pos': r.pos1,
                        'pos_english': map_pos_to_english(r.pos1),
                        'lemma': r.lemma
                    })

            elif tool == 'pos_stanza':
                from lexprep.ja.pos_stanza import tag_pretokenized_with_stanza
                tags = tag_pretokenized_with_stanza(words, download_if_missing=True)
                for i, w in enumerate(words):
                    r = tags[i]
                    results.append({
                        'word': w,
                        'pos': r.upos
                    })

        # Length tool — language-agnostic (inline)
        if tool == 'length':
            lengths = compute_length_chars(words)
            results = [{'word': w, 'length_chars': ln} for w, ln in zip(words, lengths)]

        return jsonify({'results': results, 'count': len(results)})

    except ImportError:
        logger.exception('Missing language module')
        return jsonify({'error': 'Required language module is not installed'}), 400
    except Exception:
        logger.exception('Error in process_words')
        return jsonify({'error': 'An error occurred while processing. Please try again.'}), 500


def process_persian_pos(words):
    """Process Persian POS tagging using Stanza"""
    from lexprep.fa.pos import tag_words

    results = []
    tags = tag_words(words)

    for i, w in enumerate(words):
        r = tags[i]
        results.append({
            'word': w,
            'normalized': r.normalized,
            'pos': r.pos_tag or '',
            'lemma': r.lemma or '',
            'error': r.error or ''
        })

    return results


@app.route('/api/download', methods=['POST'])
def download_results():
    """Generate downloadable ZIP from inline results"""
    try:
        data = request.json
        results = data.get('results', [])

        if not results:
            return jsonify({'error': 'No results to download'}), 400

        # ---- Sanitise user-supplied values via dict-lookup barrier ----
       
        tool = _TOOL_LOOKUP.get(data.get('tool', ''), '')
        language = _LANG_LOOKUP.get(data.get('language', ''), '')
        ext = _FMT_LOOKUP.get(data.get('format', ''), 'xlsx')

        df = pd.DataFrame(results)
        result_df = df.copy()

        added_cols = _get_added_columns(tool, result_df.columns)
        summary = _compute_summary(tool, result_df)

        ts = utc_now()
        manifest_language = None if tool == 'length' else (language or None)

        # Pre-compute ZIP filename from trusted literal values only
        manifest_tool = _manifest_tool_name(manifest_language, tool or 'unknown')
        zip_fname = make_zip_filename('lexprep_results', manifest_tool, manifest_language, ts)

        manifest = build_manifest(
            tool_key=tool or 'unknown',
            language=manifest_language,
            original_filename='inline_input.txt',
            file_type='txt',
            row_count=len(df),
            column_mapping={'word_column': 'word'},
            added_columns=added_cols,
            libraries=get_libraries_for_tool(language, tool),
            timestamp=ts,
            summary=summary,
        )
        zip_bytes, zip_name = build_zip(
            manifest=manifest,
            main_df=df,
            input_basename='lexprep_results',
            output_ext=ext,
            zip_filename=zip_fname,
        )
        return _download_bytes_response(
            zip_bytes,
            mimetype='application/zip',
            download_name=zip_name,
        )

    except Exception:
        logger.exception('Error in download_results')
        return jsonify({'error': 'An error occurred while generating the download'}), 500

@app.route('/api/parse-columns', methods=['POST'])
def parse_columns():
    """Parse uploaded file to get column names"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        ext = _safe_ext(file.filename)

        # Read file into memory
        file_bytes = io.BytesIO(file.read())

        if ext == 'csv':
            df = pd.read_csv(file_bytes, nrows=5, encoding='utf-8-sig')
        elif ext == 'tsv':
            df = pd.read_csv(file_bytes, sep='\t', nrows=5, encoding='utf-8-sig')
        elif ext in ('xlsx', 'xls'):
            df = pd.read_excel(file_bytes, nrows=5)
        else:
            return jsonify({'columns': ['word'], 'suggested': 'word'})

        columns = list(df.columns)

        # Suggest word column
        candidates = ['word', 'Word', 'WORD', 'Item', 'item', 'text', 'Text', 'token', 'Token']
        suggested = next((c for c in candidates if c in columns), columns[0] if columns else 'word')

        return jsonify({'columns': columns, 'suggested': suggested})

    except Exception:
        logger.exception('Error in parse_columns')
        return jsonify({'error': 'Could not parse file columns'}), 500


@app.route('/api/process-file', methods=['POST'])
@limiter.limit('10 per minute')
def process_file():
    """Process uploaded wordlist file"""
    import time
    start_time = time.time()

    try:
        logger.debug("process-file called")

        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        language = request.form.get('language', '')
        tool = request.form.get('tool', '')
        word_column = request.form.get('word_column', 'word')

        logger.debug("process-file: language=%s, tool=%s, word_column=%s", language, tool, word_column)

        if language not in VALID_LANGUAGES:
            return jsonify({'error': 'Invalid language'}), 400
        if tool not in VALID_TOOLS:
            return jsonify({'error': 'Invalid tool'}), 400

        ext = _safe_ext(file.filename)

        read_start = time.time()
        file_bytes = io.BytesIO(file.read())

        # Read file
        if ext == 'txt':
            content = file_bytes.read().decode('utf-8')
            words = [w.strip() for w in content.split('\n') if w.strip()]
            df = pd.DataFrame({'word': words})
            word_column = 'word'
        elif ext == 'csv':
            df = pd.read_csv(file_bytes, encoding='utf-8-sig')
        elif ext == 'tsv':
            df = pd.read_csv(file_bytes, sep='\t', encoding='utf-8-sig')
        elif ext in ('xlsx', 'xls'):
            df = pd.read_excel(file_bytes)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400

        # Validate column exists
        if word_column not in df.columns:
            return jsonify({'error': f"Column '{word_column}' not found in file"}), 400

        # Get words from dataframe
        words = df[word_column].astype(str).str.strip().tolist()
        words = [w for w in words if w and w != 'nan' and w.strip()]

        read_time = time.time() - read_start
        logger.debug("File read completed in %.2fs", read_time)

        if not words:
            return jsonify({'error': 'No words found in file'}), 400

        logger.debug("Processing %d words with %s/%s", len(words), language, tool)

        # Log usage
        log_tool_usage(language, tool, len(words))

        # Process based on language and tool
        process_start = time.time()
        results = process_words_batch(words, language, tool)
        process_time = time.time() - process_start
        logger.debug("Finished in %.2fs", process_time)

        # Add results to dataframe
        result_df = pd.DataFrame(results)

        # Create output dataframe - start with original and add result columns
        output_df = df.copy()
        for col in result_df.columns:
            if col != 'word':
                output_df[col] = result_df[col].values[:len(output_df)]

        # Build ZIP response
        out_ext = ext if ext in ('csv', 'tsv') else 'xlsx'
        total_time = time.time() - start_time
        logger.debug("Complete in %.2fs", total_time)

        response = _build_zip_response(
            tool=tool, language=language,
            word_column=word_column, df=df, output_df=output_df,
            result_df=result_df, ext=out_ext,
            original_filename=file.filename,
        )
        response.headers['X-Word-Count'] = str(len(words))
        response.headers['Access-Control-Expose-Headers'] = 'X-Word-Count, Content-Disposition'
        return response

    except ImportError:
        logger.exception("ImportError in process_file")
        return jsonify({'error': 'Required language module is not installed'}), 400
    except Exception:
        logger.exception("Error in process_file")
        return jsonify({'error': 'An error occurred while processing the file'}), 500


# ============== Async Processing API ==============

@app.route('/api/process-file-async', methods=['POST'])
@limiter.limit('10 per minute')
def process_file_async():
    """Start async file processing - returns job ID for polling"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        language = request.form.get('language', '')
        tool = request.form.get('tool', '')
        word_column = request.form.get('word_column', 'word')

        if language not in VALID_LANGUAGES:
            return jsonify({'error': 'Invalid language'}), 400
        if tool not in VALID_TOOLS:
            return jsonify({'error': 'Invalid tool'}), 400

        # Read file into memory
        file_bytes = file.read()
        ext = _safe_ext(file.filename)
        safe_name = secure_filename(file.filename) or 'upload'

        # Create job
        job_id = str(uuid.uuid4())
        with _jobs_lock:
            _jobs[job_id] = {
                'status': 'pending',
                'progress': 0,
                'message': 'Starting...',
                'result': None,
                'error': None,
                'created_at': datetime.now().isoformat()
            }

        # Start background processing
        thread = threading.Thread(
            target=_process_file_background,
            args=(job_id, file_bytes, ext, language, tool, word_column, safe_name),
            daemon=True
        )
        thread.start()

        return jsonify({
            'job_id': job_id,
            'status': 'started',
            'poll_url': f'/api/job/{job_id}'
        })

    except Exception:
        logger.exception("Error in process_file_async")
        return jsonify({'error': 'An error occurred while starting the job'}), 500


def _process_file_background(job_id, file_bytes, ext, language, tool, word_column,
                             safe_name='upload'):
    """Background processing function"""
    try:
        with _jobs_lock:
            _jobs[job_id]['status'] = 'processing'
            _jobs[job_id]['message'] = 'Reading file...'
            _jobs[job_id]['progress'] = 10

        file_io = io.BytesIO(file_bytes)

        # Read file
        if ext == 'txt':
            content = file_io.read().decode('utf-8')
            words = [w.strip() for w in content.split('\n') if w.strip()]
            df = pd.DataFrame({'word': words})
            word_column = 'word'
        elif ext == 'csv':
            df = pd.read_csv(file_io, encoding='utf-8-sig')
        elif ext == 'tsv':
            df = pd.read_csv(file_io, sep='\t', encoding='utf-8-sig')
        elif ext in ('xlsx', 'xls'):
            df = pd.read_excel(file_io)
        else:
            raise ValueError('Unsupported file format')

        with _jobs_lock:
            _jobs[job_id]['message'] = 'Extracting words...'
            _jobs[job_id]['progress'] = 20

        # Validate column exists
        if word_column not in df.columns:
            raise ValueError(f"Column '{word_column}' not found in file")

        # Get words from dataframe
        words = df[word_column].astype(str).str.strip().tolist()
        words = [w for w in words if w and w != 'nan' and w.strip()]

        if not words:
            raise ValueError('No words found in file')

        with _jobs_lock:
            _jobs[job_id]['message'] = f'Processing {len(words)} words...'
            _jobs[job_id]['progress'] = 30

        # Process words
        results = process_words_batch(words, language, tool)

        with _jobs_lock:
            _jobs[job_id]['message'] = 'Generating output...'
            _jobs[job_id]['progress'] = 80

        # Add results to dataframe
        result_df = pd.DataFrame(results)
        output_df = df.copy()
        for col in result_df.columns:
            if col != 'word':
                output_df[col] = result_df[col].values[:len(output_df)]

        # Build ZIP
        out_ext = ext if ext in ('csv', 'tsv') else 'xlsx'
        added_cols = _get_added_columns(tool, result_df.columns)
        summary = _compute_summary(tool, result_df)
        ts = utc_now()

        manifest_language = None if tool == 'length' else language
        safe_base = safe_name.rsplit('.', 1)[0] if '.' in safe_name else safe_name

        # Pre-compute ZIP filename 
        manifest_tool = _manifest_tool_name(manifest_language, tool)
        zip_fname = make_zip_filename(safe_base, manifest_tool, manifest_language, ts)

        manifest = build_manifest(
            tool_key=tool,
            language=manifest_language,
            original_filename=safe_name,
            file_type=out_ext,
            row_count=len(df),
            column_mapping={'word_column': _safe_col_name(word_column)},
            added_columns=added_cols,
            libraries=get_libraries_for_tool(language, tool),
            timestamp=ts,
            summary=summary,
        )
        zip_bytes, zip_name = build_zip(
            manifest=manifest,
            main_df=output_df,
            input_basename=safe_base,
            output_ext=out_ext,
            zip_filename=zip_fname,
        )

        with _jobs_lock:
            _jobs[job_id]['status'] = 'completed'
            _jobs[job_id]['progress'] = 100
            _jobs[job_id]['message'] = 'Complete!'
            _jobs[job_id]['result'] = {
                'data': zip_bytes,
                'mimetype': 'application/zip',
                'filename': zip_name,
                'word_count': len(words)
            }

    except Exception:
        logger.exception("Error in background job %s", job_id)
        with _jobs_lock:
            _jobs[job_id]['status'] = 'failed'
            _jobs[job_id]['error'] = 'An error occurred during processing'
            _jobs[job_id]['message'] = 'Processing failed'


@app.route('/api/job/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get job status"""
    try:
        uuid.UUID(job_id)
    except ValueError:
        return jsonify({'error': 'Invalid job ID'}), 400
    with _jobs_lock:
        if job_id not in _jobs:
            return jsonify({'error': 'Job not found'}), 404

        job = _jobs[job_id]
        return jsonify({
            'status': job['status'],
            'progress': job['progress'],
            'message': job['message'],
            'error': job['error'],
            'has_result': job['result'] is not None
        })


@app.route('/api/job/<job_id>/download', methods=['GET'])
def download_job_result(job_id):
    """Download completed job result"""
    try:
        uuid.UUID(job_id)
    except ValueError:
        return jsonify({'error': 'Invalid job ID'}), 400
    with _jobs_lock:
        if job_id not in _jobs:
            return jsonify({'error': 'Job not found'}), 404

        job = _jobs[job_id]
        if job['status'] != 'completed' or not job['result']:
            return jsonify({'error': 'Job not completed'}), 400

        result = job['result']
        response = _download_bytes_response(
            result['data'],
            mimetype=result['mimetype'],
            download_name=result['filename']
        )
        response.headers['X-Word-Count'] = str(result.get('word_count', 0))
        response.headers['Access-Control-Expose-Headers'] = 'X-Word-Count, Content-Disposition'
        return response


def process_words_batch(words, language, tool):
    import time
    results = []

    logger.debug("Processing %d words", len(words))

    if language == 'fa':
        if tool == 'g2p':
            converter = _model_cache.get('fa_g2p_converter')
            if converter:
                chunk_size = 25
                total_chunks = (len(words) - 1) // chunk_size + 1

                for i in range(0, len(words), chunk_size):
                    chunk = words[i:i+chunk_size]
                    chunk_start = time.time()
                    chunk_num = i // chunk_size + 1

                    for w in chunk:
                        word = str(w).strip()
                        if not word:
                            continue
                        try:
                            pron = converter.transliterate(word, tidy=True)
                            results.append({
                                'word': word,
                                'pronunciation': pron,
                                'error': ''
                            })
                        except Exception as e:
                            results.append({
                                'word': word,
                                'pronunciation': '',
                                'error': str(e)
                            })

                    chunk_time = time.time() - chunk_start
                    logger.debug("Chunk %d/%d done (%.2fs)", chunk_num, total_chunks, chunk_time)
            else:
                # Fall back to creating new converter (slower)
                logger.debug("No cached converter, loading fresh (this may be slow)...")
                from lexprep.fa.g2p import transcribe_words
                transcriptions = transcribe_words(words)
                for i, w in enumerate(words):
                    r = transcriptions[i] if i < len(transcriptions) else None
                    results.append({
                        'word': w,
                        'pronunciation': r.pronunciation if r else '',
                        'error': r.error or '' if r else ''
                    })
            logger.debug("Got %d transcriptions", len(results))
        elif tool == 'syllables':
            from lexprep.fa.syllables import syllabify_orthographic
            for w in words:
                syllabified, count = syllabify_orthographic(w)
                results.append({
                    'word': w,
                    'syllabified': syllabified,
                    'syllable_count': count
                })
        elif tool == 'syllables_phonetic':
            from lexprep.fa.syllables import count_syllables_from_pronunciation
            logger.debug("Persian syllables (phonetic) processing...")

            # Use cached converter if available
            converter = _model_cache.get('fa_g2p_converter')
            if converter:
                logger.debug("Using cached G2P converter...")

                # Process in chunks with progress reporting
                chunk_size = 25
                total_chunks = (len(words) - 1) // chunk_size + 1
                for i in range(0, len(words), chunk_size):
                    chunk = words[i:i+chunk_size]
                    chunk_start = time.time()
                    chunk_num = i // chunk_size + 1

                    for w in chunk:
                        word = str(w).strip()
                        if not word:
                            continue
                        try:
                            pron = converter.transliterate(word, tidy=True)
                            count = count_syllables_from_pronunciation(pron)
                            results.append({
                                'word': word,
                                'pronunciation': pron,
                                'syllable_count': count
                            })
                        except Exception:
                            results.append({
                                'word': word,
                                'pronunciation': '',
                                'syllable_count': 0
                            })

                    chunk_time = time.time() - chunk_start
                    logger.debug("Chunk %d/%d: %d words in %.2fs", chunk_num, total_chunks, len(chunk), chunk_time)
            else:
                # Fall back to creating new converter
                logger.debug("No cached converter, loading fresh...")
                from lexprep.fa.g2p import transcribe_words
                transcriptions = transcribe_words(words)
                for i, w in enumerate(words):
                    r = transcriptions[i] if i < len(transcriptions) else None
                    pron = r.pronunciation if r else ''
                    count = count_syllables_from_pronunciation(pron)
                    results.append({
                        'word': w,
                        'pronunciation': pron,
                        'syllable_count': count
                    })
            logger.debug("Processed %d words", len(results))
        elif tool == 'pos':
            results = process_persian_pos(words)

    elif language == 'en':
        if tool == 'g2p':
            from lexprep.en.g2p import transcribe_words
            transcriptions = transcribe_words(words)
            for i, w in enumerate(words):
                r = transcriptions[i] if i < len(transcriptions) else None
                results.append({
                    'word': w,
                    'pronunciation': r.pronunciation if r else '',
                    'error': r.error or '' if r else ''
                })
        elif tool == 'syllables':
            from lexprep.en.syllables import count_syllables
            for w in words:
                count = count_syllables(w)
                results.append({
                    'word': w,
                    'syllable_count': count
                })
        elif tool == 'pos':
            from lexprep.en.pos import tag_words
            tags = tag_words(words)
            for i, w in enumerate(words):
                r = tags[i] if i < len(tags) else None
                results.append({
                    'word': w,
                    'pos': r.pos if r else '',
                    'tag': r.tag if r else '',
                    'lemma': r.lemma if r else ''
                })

    elif language == 'ja':
        if tool == 'pos_unidic':
            from lexprep.ja.pos_map import map_pos_to_english
            from lexprep.ja.pos_unidic import tag_with_unidic
            tags = tag_with_unidic(words)
            for i, w in enumerate(words):
                r = tags[i] if i < len(tags) else None
                results.append({
                    'word': w,
                    'pos': r.pos1 if r else '',
                    'pos_english': map_pos_to_english(r.pos1) if r else '',
                    'lemma': r.lemma if r else ''
                })
        elif tool == 'pos_stanza':
            from lexprep.ja.pos_stanza import tag_pretokenized_with_stanza
            tags = tag_pretokenized_with_stanza(words, download_if_missing=True)
            for i, w in enumerate(words):
                r = tags[i] if i < len(tags) else None
                results.append({
                    'word': w,
                    'pos': r.upos if r else ''
                })

    # Length tool — language-agnostic
    if tool == 'length':
        lengths = compute_length_chars(words)
        results = []
        for w, ln in zip(words, lengths):
            results.append({'word': w, 'length_chars': ln})

    return results


# ============== Admin API ==============

@app.route('/api/admin/stats', methods=['GET'])
@admin_required
def admin_stats():
    """Get analytics statistics"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()

            # Total page views
            c.execute('SELECT COUNT(*) FROM page_views')
            total_views = c.fetchone()[0]

            # Views per page
            c.execute('SELECT page, COUNT(*) as count FROM page_views GROUP BY page ORDER BY count DESC')
            views_by_page = dict(c.fetchall())

            # Unique visitors (by IP hash)
            c.execute('SELECT COUNT(DISTINCT ip_hash) FROM page_views')
            unique_visitors = c.fetchone()[0]

            # Tool usage stats
            c.execute('SELECT language, tool, COUNT(*) as count, SUM(word_count) as total_words FROM tool_usage GROUP BY language, tool ORDER BY count DESC')
            tool_stats = []
            for row in c.fetchall():
                tool_stats.append({
                    'language': row[0],
                    'tool': row[1],
                    'usage_count': row[2],
                    'total_words': row[3]
                })

            # Recent activity (last 7 days)
            c.execute('''SELECT DATE(timestamp) as date, COUNT(*) as count
                         FROM page_views
                         WHERE timestamp >= datetime('now', '-7 days')
                         GROUP BY DATE(timestamp)
                         ORDER BY date DESC''')
            daily_views = [{'date': row[0], 'views': row[1]} for row in c.fetchall()]

            # Daily tool usage (last 7 days)
            c.execute('''SELECT DATE(timestamp) as date,
                                COUNT(*) as usage_count,
                                COALESCE(SUM(word_count), 0) as total_words
                         FROM tool_usage
                         WHERE timestamp >= datetime('now', '-7 days')
                         GROUP BY DATE(timestamp)
                         ORDER BY date DESC''')
            daily_tool_usage = [
                {'date': row[0], 'usage_count': row[1], 'total_words': row[2]}
                for row in c.fetchall()
            ]

        return jsonify({
            'total_views': total_views,
            'unique_visitors': unique_visitors,
            'views_by_page': views_by_page,
            'tool_stats': tool_stats,
            'daily_views': daily_views,
            'daily_tool_usage': daily_tool_usage,
        })

    except Exception:
        logger.exception('Error in admin_stats')
        return jsonify({'error': 'An internal error occurred'}), 500


@app.route('/api/track', methods=['POST'])
def track_event():
    """Track custom events"""
    try:
        data = request.json
        page = data.get('page', 'unknown')
        log_page_view(page)
        return jsonify({'status': 'ok'})
    except Exception:
        logger.exception('Failed to track event')
        return jsonify({'status': 'ok'})


# ============== Sampling API ==============

@app.route('/api/sampling/parse-file', methods=['POST'])
@limiter.limit('20 per minute')
def sampling_parse_file():
    """Parse uploaded file and return columns with numeric detection"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if not file.filename:
            return jsonify({'error': 'No file selected'}), 400

        ext = _safe_ext(file.filename)
        file_bytes = io.BytesIO(file.read())

        try:
            if ext == 'csv':
                df = pd.read_csv(file_bytes, encoding='utf-8-sig')
            elif ext == 'tsv':
                df = pd.read_csv(file_bytes, sep='\t', encoding='utf-8-sig')
            elif ext in ('xlsx', 'xls'):
                df = pd.read_excel(file_bytes)
            else:
                return jsonify({'error': 'Unsupported file format. Use CSV, TSV, or Excel files.'}), 400
        except Exception:
            logger.exception('Could not read file in sampling_parse_file')
            return jsonify({'error': 'Could not read file'}), 400

        if len(df.columns) == 0:
            return jsonify({'error': 'File appears to be empty or has no columns'}), 400

        # Detect numeric columns
        columns = []
        for col in df.columns:
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            # Also check if convertible to numeric
            if not is_numeric:
                converted = pd.to_numeric(df[col], errors='coerce')
                is_numeric = bool(converted.notna().sum() > len(df) * 0.5)  # >50% numeric

            # Get sample values, converting to JSON-safe format
            sample_vals = df[col].head(3).tolist()
            # Handle NaN and other non-JSON values
            safe_samples = []
            for v in sample_vals:
                if pd.isna(v):
                    safe_samples.append(None)
                elif isinstance(v, (int, float, str, bool)):
                    safe_samples.append(v)
                else:
                    safe_samples.append(str(v))

            columns.append({
                'name': str(col),
                'is_numeric': bool(is_numeric),
                'sample_values': safe_samples
            })

        # Suggest stratification column (first numeric column with "freq" or similar)
        suggested = None
        priority_keywords = ['freq', 'frequency', 'count', 'score', 'value', 'rate']
        for col in columns:
            if col['is_numeric']:
                col_lower = col['name'].lower()
                for kw in priority_keywords:
                    if kw in col_lower:
                        suggested = col['name']
                        break
                if suggested:
                    break

        # Fall back to first numeric column
        if not suggested:
            for col in columns:
                if col['is_numeric']:
                    suggested = col['name']
                    break

        return jsonify({
            'columns': columns,
            'suggested_column': suggested,
            'row_count': int(len(df)),
            'filename': secure_filename(file.filename) or 'upload'
        })

    except Exception:
        logger.exception('Error in sampling_parse_file')
        return jsonify({'error': 'An error occurred while processing the file'}), 500


@app.route('/api/sampling/stratified', methods=['POST'])
@limiter.limit('10 per minute')
def sampling_stratified():
    """Stratified sampling - supports both quantile and custom range modes"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        score_col = request.form.get('score_col', '')
        n_total = int(request.form.get('n_total', 0))
        mode = request.form.get('mode', 'quantile')  # 'quantile' or 'custom'
        allocation = request.form.get('allocation', 'equal')
        random_state = int(request.form.get('random_state', 19))

        # Parse weights if provided
        weights_json = request.form.get('weights', '')
        weights = None
        if weights_json:
            try:
                weights = json.loads(weights_json)
                weights = {int(k): float(v) for k, v in weights.items()}
            except (json.JSONDecodeError, ValueError):
                weights = None

        if not score_col:
            return jsonify({'error': 'Stratification column is required'}), 400

        # Read file
        ext = _safe_ext(file.filename)
        safe_name = secure_filename(file.filename) or 'upload'
        safe_base = safe_name.rsplit('.', 1)[0] if '.' in safe_name else safe_name
        file_bytes = io.BytesIO(file.read())

        if ext == 'csv':
            df = pd.read_csv(file_bytes, encoding='utf-8-sig')
        elif ext == 'tsv':
            df = pd.read_csv(file_bytes, sep='\t', encoding='utf-8-sig')
        elif ext in ('xlsx', 'xls'):
            df = pd.read_excel(file_bytes)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400

        if mode == 'custom':
            # Custom range mode
            ranges_json = request.form.get('ranges', '[]')
            try:
                ranges_data = json.loads(ranges_json)
            except json.JSONDecodeError:
                return jsonify({'error': 'Invalid ranges format'}), 400

            from lexprep.sampling.stratified import (
                CustomRange,
                stratified_sample_custom_ranges_full,
            )

            ranges = []
            for r in ranges_data:
                ranges.append(CustomRange(
                    lower=float(r.get('lower', float('-inf'))),
                    upper=float(r.get('upper', float('inf'))),
                    lower_inclusive=r.get('lower_inclusive', True),
                    upper_inclusive=r.get('upper_inclusive', True),
                    label=r.get('label')
                ))

            # Parse fixed counts if provided
            fixed_counts = None
            fixed_counts_json = request.form.get('fixed_counts', '')
            if fixed_counts_json and allocation == 'fixed':
                try:
                    fixed_counts = json.loads(fixed_counts_json)
                    fixed_counts = {int(k): int(v) for k, v in fixed_counts.items()}
                except (json.JSONDecodeError, ValueError):
                    fixed_counts = None

            result = stratified_sample_custom_ranges_full(
                df,
                score_col=score_col,
                ranges=ranges,
                n_total=n_total if allocation != 'fixed' else None,
                allocation=allocation,
                weights=weights,
                fixed_counts=fixed_counts,
                random_state=random_state
            )
        else:
            # Quantile mode
            bins = int(request.form.get('bins', 3))

            if n_total <= 0:
                return jsonify({'error': 'Total samples must be greater than 0'}), 400

            from lexprep.sampling.stratified import stratified_sample_quantiles_full

            result = stratified_sample_quantiles_full(
                df,
                score_col=score_col,
                n_total=n_total,
                bins=bins,
                allocation=allocation,
                weights=weights,
                random_state=random_state
            )

        # Build ZIP with sample + excluded + audit + manifest
        from lexprep.sampling.audit import audit_to_bytes, build_sampling_manifest_section

        report = result.report
        audit_bytes_data = audit_to_bytes(report)
        sampling_section = build_sampling_manifest_section(report)

        ts = utc_now()

        # Pre-compute ZIP filename from trusted
        
        zip_fname = make_zip_filename(
            safe_base, 'stratified_sampling', None, ts,
        )

        manifest = build_manifest(
            tool_key='stratified',
            language=None,
            original_filename=safe_name,
            file_type=ext if ext in ('csv', 'tsv') else 'xlsx',
            row_count=len(df),
            column_mapping={'score_column': _safe_col_name(score_col)},
            added_columns=['bin_id'],
            libraries=[],
            timestamp=ts,
            reproducibility={'seed': random_state, 'parameters': {
                'mode': 'quantiles' if mode == 'quantile' else 'custom_ranges',
                'bins': int(request.form.get('bins', 3)) if mode == 'quantile' else len(ranges),
                'n_total': n_total,
            }},
            sampling=sampling_section,
        )
        zip_bytes, zip_name = build_zip(
            manifest=manifest,
            main_df=result.sample_df,
            input_basename=safe_base,
            output_ext='xlsx',
            is_sampling=True,
            excluded_df=result.excluded_df,
            audit_bytes=audit_bytes_data,
            zip_filename=zip_fname,
        )

        response = _download_bytes_response(
            zip_bytes,
            mimetype='application/zip',
            download_name=zip_name,
        )

        if report.warnings:
            response.headers['X-LexPrep-Warnings'] = json.dumps(report.warnings)

        return response

    except ImportError:
        logger.exception('Sampling module not installed')
        return jsonify({'error': 'Sampling module not installed'}), 400
    except Exception:
        logger.exception('Error in sampling_stratified')
        return jsonify({'error': 'An error occurred during sampling'}), 500


@app.route('/api/sampling/shuffle', methods=['POST'])
@limiter.limit('10 per minute')
def sampling_shuffle():
    """Shuffle rows across multiple files - returns ZIP with all shuffled files"""
    try:
        files = request.files.getlist('files')
        if len(files) < 2:
            return jsonify({'error': 'At least 2 files required'}), 400

        seed = int(request.form.get('seed', 19))
        n_columns = request.form.get('n_columns')
        n_columns = int(n_columns) if n_columns and n_columns.strip() else None

        # Read all files
        dfs = []
        exts = []
        safe_names = []
        for file in files:
            ext = _safe_ext(file.filename)
            exts.append(ext)
            safe_names.append(secure_filename(file.filename) or f'file_{len(exts)}')
            file_bytes = io.BytesIO(file.read())

            if ext == 'csv':
                df = pd.read_csv(file_bytes, encoding='utf-8-sig')
            elif ext == 'tsv':
                df = pd.read_csv(file_bytes, sep='\t', encoding='utf-8-sig')
            elif ext in ('xlsx', 'xls'):
                df = pd.read_excel(file_bytes)
            else:
                return jsonify({'error': 'Unsupported file format'}), 400

            dfs.append(df)

        # Perform shuffling
        from lexprep.sampling.shuffle_rows import shuffle_corresponding_rows

        out_dfs, shuffle_report = shuffle_corresponding_rows(
            dfs,
            seed=seed,
            n_columns=n_columns
        )

        # Build extra files for ZIP
        extra_files = []
        for out_df, sn, file_ext in zip(out_dfs, safe_names, exts):
            out_ext = file_ext if file_ext in ('csv', 'tsv') else 'xlsx'
            base = sn.rsplit('.', 1)[0] if '.' in sn else sn
            out_filename = f'{base}_shuffled.{out_ext}'
            extra_files.append((out_filename, _df_to_bytes(out_df, out_ext)))

        ts = utc_now()

        # Pre-compute ZIP filename from trusted values (breaks taint chain)
        zip_fname = make_zip_filename('shuffle', 'row_shuffle', None, ts)

        manifest = build_manifest(
            tool_key='shuffle',
            language=None,
            original_filename=', '.join(safe_names),
            file_type='multiple',
            row_count=shuffle_report.n_rows,
            column_mapping={'used_columns': shuffle_report.used_columns},
            added_columns=[],
            libraries=[],
            timestamp=ts,
            reproducibility={
                'seed': seed,
                'parameters': {
                    'number_of_files': shuffle_report.n_files,
                    'row_count': shuffle_report.n_rows,
                    'shuffle_mode': 'synchronized_row_permutation',
                },
            },
        )
        zip_bytes, zip_name = build_zip(
            manifest=manifest,
            main_df=pd.DataFrame(),
            input_basename='shuffle',
            output_ext='xlsx',
            extra_files=extra_files,
            zip_filename=zip_fname,
        )

        return _download_bytes_response(
            zip_bytes,
            mimetype='application/zip',
            download_name=zip_name,
        )

    except ImportError:
        return jsonify({'error': 'Sampling module not installed'}), 400
    except Exception:
        logger.exception('Error in sampling_shuffle')
        return jsonify({'error': 'An error occurred during shuffle'}), 500


# ============== Security Headers ==============

@app.after_request
def set_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['X-XSS-Protection'] = '0'
    return response


if __name__ == '__main__':
    # Auto-warmup in development mode
    import sys

    # Force unbuffered output for immediate terminal logging
    sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

    if '--warmup' in sys.argv or os.environ.get('LEXPREP_WARMUP', '').lower() == 'true':
        print("[Startup] Running model warmup...")
        sys.stdout.flush()
        preload_models()

    print("[Server] Starting Flask on http://127.0.0.1:5000")
    print("[Server] Logs will appear below when requests are received")
    sys.stdout.flush()
    app.run(
        debug=os.environ.get('FLASK_DEBUG', 'false').lower() == 'true',
        port=5000,
        use_reloader=os.environ.get('FLASK_DEBUG', 'false').lower() == 'true',
    )
