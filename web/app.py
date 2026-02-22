from flask import Flask, request, jsonify, send_file, render_template
from markupsafe import Markup
from flask_cors import CORS
import pandas as pd
import io
import json
import os
import uuid
import threading
from datetime import datetime
from functools import wraps
from pathlib import Path
import hashlib
import hmac
import sqlite3
import logging

from lexprep.manifest import build_manifest, get_libraries_for_tool, utc_now
from lexprep.packaging import build_zip, _df_to_bytes
from lexprep.length import LENGTH_METHOD, compute_length_chars, length_distribution

logger = logging.getLogger(__name__)

# Load environment variables 
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:

    pass

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['JSON_AS_ASCII'] = False  
CORS(app)

# Version
VERSION = '1.0.0'


ADMIN_SECRET = os.environ.get('LEXPREP_ADMIN_SECRET')
ADMIN_ENABLED = ADMIN_SECRET is not None and len(ADMIN_SECRET) >= 32

# Analytics database path
DB_PATH = os.path.join(os.path.dirname(__file__), 'analytics.db')


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
        _warmup_status['errors'].append(f'fa_g2p: {str(e)}')
        print(f"[Warmup] Persian G2P failed: {e}")

    # Persian POS (Stanza)
    try:
        from lexprep.fa.pos import tag_words as fa_tag_words
        fa_tag_words(['سلام'])
        _model_cache['fa_pos'] = True
        _warmup_status['loaded'].append('fa_pos')
        print("[Warmup] Persian POS loaded (Stanza)")
    except Exception as e:
        _warmup_status['errors'].append(f'fa_pos: {str(e)}')
        print(f"[Warmup] Persian POS failed: {e}")

    # English G2P
    try:
        from lexprep.en.g2p import transcribe_words
        transcribe_words(['hello'])
        _model_cache['en_g2p'] = True
        _warmup_status['loaded'].append('en_g2p')
        print("[Warmup] English G2P loaded")
    except Exception as e:
        _warmup_status['errors'].append(f'en_g2p: {str(e)}')
        print(f"[Warmup] English G2P failed: {e}")

    # English POS (spaCy)
    try:
        from lexprep.en.pos import tag_words
        tag_words(['hello'])
        _model_cache['en_pos'] = True
        _warmup_status['loaded'].append('en_pos')
        print("[Warmup] English POS loaded")
    except Exception as e:
        _warmup_status['errors'].append(f'en_pos: {str(e)}')
        print(f"[Warmup] English POS failed: {e}")

    # Japanese UniDic
    try:
        from lexprep.ja.pos_unidic import tag_with_unidic
        tag_with_unidic(['日本語'])
        _model_cache['ja_unidic'] = True
        _warmup_status['loaded'].append('ja_unidic')
        print("[Warmup] Japanese UniDic loaded")
    except Exception as e:
        _warmup_status['errors'].append(f'ja_unidic: {str(e)}')
        print(f"[Warmup] Japanese UniDic failed: {e}")

    # Japanese Stanza (optional - can be slow)
    try:
        from lexprep.ja.pos_stanza import tag_pretokenized_with_stanza
        tag_pretokenized_with_stanza(['日本語'], download_if_missing=True)
        _model_cache['ja_stanza'] = True
        _warmup_status['loaded'].append('ja_stanza')
        print("[Warmup] Japanese Stanza loaded")
    except Exception as e:
        _warmup_status['errors'].append(f'ja_stanza: {str(e)}')
        print(f"[Warmup] Japanese Stanza failed: {e}")

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


def init_db():
    """Initialize SQLite database for analytics"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS page_views (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        page TEXT,
        ip_hash TEXT,
        user_agent TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS tool_usage (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        language TEXT,
        tool TEXT,
        word_count INTEGER,
        ip_hash TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()
    conn.close()


def get_client_ip():
    """Get client IP, checking X-Forwarded-For for proxied requests."""
    forwarded = request.headers.get('X-Forwarded-For', '')
    if forwarded:
        # X-Forwarded-For can be a comma-separated list; first is the real client
        return forwarded.split(',')[0].strip()
    return request.remote_addr or 'unknown'


def hash_ip(ip):
    """Hash IP for privacy"""
    return hashlib.sha256(ip.encode()).hexdigest()[:16]


def log_page_view(page):
    """Log a page view"""
    try:
        ip = get_client_ip()
        ua = request.headers.get('User-Agent', '')[:200]
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('INSERT INTO page_views (page, ip_hash, user_agent) VALUES (?, ?, ?)',
                  (page, hash_ip(ip), ua))
        conn.commit()
        conn.close()
    except Exception:
        logger.exception('Failed to log page view for page: %s', page)


def log_tool_usage(language, tool, word_count):
    """Log tool usage"""
    try:
        ip = get_client_ip()
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('INSERT INTO tool_usage (language, tool, word_count, ip_hash) VALUES (?, ?, ?, ?)',
                  (language, tool, word_count, hash_ip(ip)))
        conn.commit()
        conn.close()
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


def _get_output_ext(filename):
    """Determine output file extension from input filename."""
    ext = Path(filename).suffix.lstrip('.').lower()
    return ext if ext in ('csv', 'tsv') else 'xlsx'


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
    tool, language, filename, word_column, df, output_df, result_df,
    ext=None, timestamp=None,
):
    """Build a ZIP response for language tool endpoints."""
    ts = timestamp or utc_now()
    ext = ext or _get_output_ext(filename)
    input_basename = Path(filename).stem

    # Language-agnostic tools should not record a language
    manifest_language = None if tool == 'length' else language

    added_cols = _get_added_columns(tool, result_df.columns)
    summary = _compute_summary(tool, result_df)

    manifest = build_manifest(
        tool_key=tool,
        language=manifest_language,
        original_filename=filename,
        file_type=Path(filename).suffix.lstrip('.'),
        row_count=len(df),
        column_mapping={'word_column': word_column},
        added_columns=added_cols,
        libraries=get_libraries_for_tool(language, tool),
        timestamp=ts,
        summary=summary,
    )
    zip_bytes, zip_name = build_zip(
        manifest=manifest,
        main_df=output_df,
        input_basename=input_basename,
        output_ext=ext,
    )
    return send_file(
        io.BytesIO(zip_bytes),
        mimetype='application/zip',
        as_attachment=True,
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
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500




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
            'errors': _warmup_status['errors']
        }
    }

    try:
        from lexprep.fa.g2p import transcribe_words
        status['modules']['fa'] = 'ok'
    except ImportError as e:
        status['modules']['fa'] = str(e)

    try:
        from lexprep.en.g2p import transcribe_words
        status['modules']['en'] = 'ok'
    except ImportError as e:
        status['modules']['en'] = str(e)

    try:
        from lexprep.ja.pos_unidic import tag_with_unidic
        status['modules']['ja'] = 'ok'
    except ImportError as e:
        status['modules']['ja'] = str(e)

    return jsonify(status)


@app.route('/api/warmup', methods=['POST'])
def warmup():
    """Trigger model warmup - call this after server starts"""
    if _warmup_status['started']:
        return jsonify({
            'status': 'already_started',
            'completed': _warmup_status['completed'],
            'loaded': _warmup_status['loaded'],
            'errors': _warmup_status['errors']
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
        'errors': _warmup_status['errors']
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

        if not language or not tool:
            return jsonify({'error': 'Language and tool are required'}), 400

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
                from lexprep.ja.pos_unidic import tag_with_unidic
                from lexprep.ja.pos_map import map_pos_to_english
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

    except ImportError as e:
        return jsonify({'error': f'Language module not installed: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
        format_type = data.get('format', 'csv')
        language = data.get('language', '')
        tool = data.get('tool', '')

        if not results:
            return jsonify({'error': 'No results to download'}), 400

        df = pd.DataFrame(results)
        result_df = df.copy()
        ext = format_type if format_type in ('csv', 'tsv') else 'xlsx'

        added_cols = _get_added_columns(tool, result_df.columns)
        summary = _compute_summary(tool, result_df)

        ts = utc_now()
        manifest_language = None if tool == 'length' else (language or None)
        manifest = build_manifest(
            tool_key=tool,
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
        )
        return send_file(
            io.BytesIO(zip_bytes),
            mimetype='application/zip',
            as_attachment=True,
            download_name=zip_name,
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============== File Processing API ==============

@app.route('/api/parse-columns', methods=['POST'])
def parse_columns():
    """Parse uploaded file to get column names"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        filename = file.filename.lower()

        # Read file into memory
        file_bytes = io.BytesIO(file.read())

        if filename.endswith('.csv'):
            df = pd.read_csv(file_bytes, nrows=5, encoding='utf-8-sig')
        elif filename.endswith('.tsv'):
            df = pd.read_csv(file_bytes, sep='\t', nrows=5, encoding='utf-8-sig')
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_bytes, nrows=5)
        else:
            return jsonify({'columns': ['word'], 'suggested': 'word'})

        columns = list(df.columns)

        # Suggest word column
        candidates = ['word', 'Word', 'WORD', 'Item', 'item', 'text', 'Text', 'token', 'Token']
        suggested = next((c for c in candidates if c in columns), columns[0] if columns else 'word')

        return jsonify({'columns': columns, 'suggested': suggested})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/process-file', methods=['POST'])
def process_file():
    """Process uploaded wordlist file"""
    import sys
    import time
    start_time = time.time()

    try:
        print(f"[API] process-file called", file=sys.stderr, flush=True)

        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        language = request.form.get('language', '')
        tool = request.form.get('tool', '')
        word_column = request.form.get('word_column', 'word')

        print(f"[API] Language: {language}, Tool: {tool}, Cached models: fa_g2p={_model_cache.get('fa_g2p_converter') is not None}", file=sys.stderr, flush=True)
        print(f"[API] language={language}, tool={tool}, word_column={word_column}", file=sys.stderr, flush=True)

        if not language or not tool:
            return jsonify({'error': 'Language and tool are required'}), 400

        filename = file.filename.lower()

        read_start = time.time()
        file_bytes = io.BytesIO(file.read())

        # Read file
        if filename.endswith('.txt'):
            content = file_bytes.read().decode('utf-8')
            words = [w.strip() for w in content.split('\n') if w.strip()]
            df = pd.DataFrame({'word': words})
            word_column = 'word'
        elif filename.endswith('.csv'):
            df = pd.read_csv(file_bytes, encoding='utf-8-sig')
        elif filename.endswith('.tsv'):
            df = pd.read_csv(file_bytes, sep='\t', encoding='utf-8-sig')
        elif filename.endswith(('.xlsx', '.xls')):
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
        print(f"[API] File read completed in {read_time:.2f}s", file=sys.stderr, flush=True)

        if not words:
            return jsonify({'error': 'No words found in file'}), 400

        print(f"[API] Processing {len(words)} words with {language}/{tool}", file=sys.stderr, flush=True)

        # Log usage
        log_tool_usage(language, tool, len(words))

        # Process based on language and tool
        process_start = time.time()
        results = process_words_batch(words, language, tool)
        process_time = time.time() - process_start
        print(f"[API] Finished in {process_time:.2f}s", file=sys.stderr, flush=True)

        # Add results to dataframe
        result_df = pd.DataFrame(results)

        # Create output dataframe - start with original and add result columns
        output_df = df.copy()
        for col in result_df.columns:
            if col != 'word':
                output_df[col] = result_df[col].values[:len(output_df)]

        # Build ZIP response
        ext = _get_output_ext(filename)
        total_time = time.time() - start_time
        print(f"[API] Complete in {total_time:.2f}s", file=sys.stderr, flush=True)

        response = _build_zip_response(
            tool=tool, language=language, filename=file.filename,
            word_column=word_column, df=df, output_df=output_df,
            result_df=result_df, ext=ext,
        )
        response.headers['X-Word-Count'] = str(len(words))
        response.headers['Access-Control-Expose-Headers'] = 'X-Word-Count, Content-Disposition'
        return response

    except ImportError as e:
        print(f"[API] ImportError: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return jsonify({'error': f'Language module not installed: {str(e)}'}), 400
    except Exception as e:
        print(f"[API] Exception: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return jsonify({'error': str(e)}), 500


# ============== Async Processing API ==============

@app.route('/api/process-file-async', methods=['POST'])
def process_file_async():
    """Start async file processing - returns job ID for polling"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        language = request.form.get('language', '')
        tool = request.form.get('tool', '')
        word_column = request.form.get('word_column', 'word')

        if not language or not tool:
            return jsonify({'error': 'Language and tool are required'}), 400

        # Read file into memory
        file_bytes = file.read()
        filename = file.filename.lower()

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
            args=(job_id, file_bytes, filename, language, tool, word_column),
            daemon=True
        )
        thread.start()

        return jsonify({
            'job_id': job_id,
            'status': 'started',
            'poll_url': f'/api/job/{job_id}'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _process_file_background(job_id, file_bytes, filename, language, tool, word_column):
    """Background processing function"""
    try:
        with _jobs_lock:
            _jobs[job_id]['status'] = 'processing'
            _jobs[job_id]['message'] = 'Reading file...'
            _jobs[job_id]['progress'] = 10

        file_io = io.BytesIO(file_bytes)

        # Read file
        if filename.endswith('.txt'):
            content = file_io.read().decode('utf-8')
            words = [w.strip() for w in content.split('\n') if w.strip()]
            df = pd.DataFrame({'word': words})
            word_column = 'word'
        elif filename.endswith('.csv'):
            df = pd.read_csv(file_io, encoding='utf-8-sig')
        elif filename.endswith('.tsv'):
            df = pd.read_csv(file_io, sep='\t', encoding='utf-8-sig')
        elif filename.endswith(('.xlsx', '.xls')):
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
        ext = _get_output_ext(filename)
        input_basename = Path(filename).stem
        added_cols = _get_added_columns(tool, result_df.columns)
        summary = _compute_summary(tool, result_df)
        ts = utc_now()

        manifest_language = None if tool == 'length' else language
        manifest = build_manifest(
            tool_key=tool,
            language=manifest_language,
            original_filename=filename,
            file_type=Path(filename).suffix.lstrip('.'),
            row_count=len(df),
            column_mapping={'word_column': word_column},
            added_columns=added_cols,
            libraries=get_libraries_for_tool(language, tool),
            timestamp=ts,
            summary=summary,
        )
        zip_bytes, zip_name = build_zip(
            manifest=manifest,
            main_df=output_df,
            input_basename=input_basename,
            output_ext=ext,
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

    except Exception as e:
        with _jobs_lock:
            _jobs[job_id]['status'] = 'failed'
            _jobs[job_id]['error'] = str(e)
            _jobs[job_id]['message'] = f'Error: {str(e)}'


@app.route('/api/job/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get job status"""
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
    with _jobs_lock:
        if job_id not in _jobs:
            return jsonify({'error': 'Job not found'}), 404

        job = _jobs[job_id]
        if job['status'] != 'completed' or not job['result']:
            return jsonify({'error': 'Job not completed'}), 400

        result = job['result']
        output = io.BytesIO(result['data'])

        response = send_file(
            output,
            mimetype=result['mimetype'],
            as_attachment=True,
            download_name=result['filename']
        )
        response.headers['X-Word-Count'] = str(result.get('word_count', 0))
        response.headers['Access-Control-Expose-Headers'] = 'X-Word-Count, Content-Disposition'
        return response


def process_words_batch(words, language, tool):
    import sys
    import time
    results = []

    print(f"Processing {len(words)} words", file=sys.stderr, flush=True)

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
                    print(f"Chunk {chunk_num}/{total_chunks} done ({chunk_time:.2f}s)", file=sys.stderr, flush=True)
            else:
                # Fall back to creating new converter (slower)
                print(f"[Batch] No cached converter, loading fresh (this may be slow)...", file=sys.stderr, flush=True)
                from lexprep.fa.g2p import transcribe_words
                transcriptions = transcribe_words(words)
                for i, w in enumerate(words):
                    r = transcriptions[i] if i < len(transcriptions) else None
                    results.append({
                        'word': w,
                        'pronunciation': r.pronunciation if r else '',
                        'error': r.error or '' if r else ''
                    })
            print(f"[Batch] Got {len(results)} transcriptions", file=sys.stderr, flush=True)
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
            print(f"[Batch] Persian syllables (phonetic) processing...", file=sys.stderr, flush=True)

            # Use cached converter if available
            converter = _model_cache.get('fa_g2p_converter')
            if converter:
                print(f"[Batch] Using cached G2P converter...", file=sys.stderr, flush=True)

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
                        except Exception as e:
                            results.append({
                                'word': word,
                                'pronunciation': '',
                                'syllable_count': 0
                            })

                    chunk_time = time.time() - chunk_start
                    print(f"[Batch] Chunk {chunk_num}/{total_chunks}: {len(chunk)} words in {chunk_time:.2f}s", file=sys.stderr, flush=True)
            else:
                # Fall back to creating new converter
                print(f"[Batch] No cached converter, loading fresh...", file=sys.stderr, flush=True)
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
            print(f"[Batch] Processed {len(results)} words", file=sys.stderr, flush=True)
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
            from lexprep.ja.pos_unidic import tag_with_unidic
            from lexprep.ja.pos_map import map_pos_to_english
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
        conn = sqlite3.connect(DB_PATH)
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

        conn.close()

        return jsonify({
            'total_views': total_views,
            'unique_visitors': unique_visitors,
            'views_by_page': views_by_page,
            'tool_stats': tool_stats,
            'daily_views': daily_views
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
def sampling_parse_file():
    """Parse uploaded file and return columns with numeric detection"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if not file.filename:
            return jsonify({'error': 'No file selected'}), 400

        filename = file.filename.lower()
        file_bytes = io.BytesIO(file.read())

        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(file_bytes, encoding='utf-8-sig')
            elif filename.endswith('.tsv'):
                df = pd.read_csv(file_bytes, sep='\t', encoding='utf-8-sig')
            elif filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_bytes)
            else:
                return jsonify({'error': 'Unsupported file format. Use CSV, TSV, or Excel files.'}), 400
        except Exception as read_error:
            return jsonify({'error': f'Could not read file: {str(read_error)}'}), 400

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
            'filename': file.filename
        })

    except Exception as e:
        import traceback
        traceback.print_exc()  # Print to server logs for debugging
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500


@app.route('/api/sampling/stratified', methods=['POST'])
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
        filename = file.filename
        file_bytes = io.BytesIO(file.read())
        filename_lower = filename.lower()

        if filename_lower.endswith('.csv'):
            df = pd.read_csv(file_bytes, encoding='utf-8-sig')
        elif filename_lower.endswith('.tsv'):
            df = pd.read_csv(file_bytes, sep='\t', encoding='utf-8-sig')
        elif filename_lower.endswith(('.xlsx', '.xls')):
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

            from lexprep.sampling.stratified import stratified_sample_custom_ranges_full, CustomRange

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
        input_basename = Path(filename).stem
        manifest = build_manifest(
            tool_key='stratified',
            language=None,
            original_filename=filename,
            file_type=Path(filename).suffix.lstrip('.'),
            row_count=len(df),
            column_mapping={'score_column': score_col},
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
            input_basename=input_basename,
            output_ext='xlsx',
            is_sampling=True,
            excluded_df=result.excluded_df,
            audit_bytes=audit_bytes_data,
        )

        response = send_file(
            io.BytesIO(zip_bytes),
            mimetype='application/zip',
            as_attachment=True,
            download_name=zip_name,
        )

        if report.warnings:
            response.headers['X-LexPrep-Warnings'] = json.dumps(report.warnings)

        return response

    except ImportError as e:
        return jsonify({'error': f'Sampling module not installed: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/sampling/shuffle', methods=['POST'])
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
        filenames = []
        for file in files:
            filename_lower = file.filename.lower()
            filenames.append(file.filename)
            file_bytes = io.BytesIO(file.read())

            if filename_lower.endswith('.csv'):
                df = pd.read_csv(file_bytes, encoding='utf-8-sig')
            elif filename_lower.endswith('.tsv'):
                df = pd.read_csv(file_bytes, sep='\t', encoding='utf-8-sig')
            elif filename_lower.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_bytes)
            else:
                return jsonify({'error': f'Unsupported file format: {file.filename}'}), 400

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
        for out_df, original_name in zip(out_dfs, filenames):
            base_name = original_name.rsplit('.', 1)[0] if '.' in original_name else original_name
            ext = original_name.rsplit('.', 1)[1].lower() if '.' in original_name else 'xlsx'
            if ext not in ('csv', 'tsv'):
                ext = 'xlsx'
            out_filename = f'{base_name}_shuffled.{ext}'
            extra_files.append((out_filename, _df_to_bytes(out_df, ext)))

        ts = utc_now()
        manifest = build_manifest(
            tool_key='shuffle',
            language=None,
            original_filename=', '.join(filenames),
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
        )

        return send_file(
            io.BytesIO(zip_bytes),
            mimetype='application/zip',
            as_attachment=True,
            download_name=zip_name,
        )

    except ImportError:
        return jsonify({'error': 'Sampling module not installed. Install lexprep with: pip install lexprep'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
    app.run(debug=True, port=5000, use_reloader=True)