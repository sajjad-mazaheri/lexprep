# Technical Documentation

This document explains how each processing method works in lexprep.

---

## English Tools

### G2P (Grapheme-to-Phoneme) - `lexprep en g2p`

**Library**: [g2p_en](https://github.com/Kyubyong/g2p) by Kyubyong Park

**How it works**:
1. Spells out numbers and currency symbols (e.g., "$200" -> "two hundred dollars")
2. Disambiguates heteronyms using POS tagging (e.g., "refuse" as verb vs noun)
3. Looks up [CMU Pronouncing Dictionary](http://www.speech.cs.cmu.edu/cgi-bin/cmudict) for known words
4. Uses a neural network model for out-of-vocabulary (OOV) words

**Output format**: ARPAbet phonemes with stress markers (0, 1, 2)
- Example: "hello" -> `HH AH0 L OW1`

**Reference**:
```bibtex
@misc{g2pE2019,
  author = {Park, Kyubyong & Kim, Jongseok},
  title = {g2pE},
  year = {2019},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Kyubyong/g2p}}
}
```

---

### Syllable Counting - `lexprep en syllables`

**Library**: [pyphen](https://github.com/Kozea/Pyphen)

**How it works**:
1. Uses hyphenation dictionaries based on TeX patterns
2. Hyphenates word at syllable boundaries (e.g., "computer" -> "com-pu-ter")
3. Counts hyphen positions + 1 = syllable count

**Dictionary**: English (US) hyphenation patterns from [hyph-utf8](http://www.hyphenation.org)

**Limitations**:
- Proper nouns may not hyphenate correctly
- Very rare or technical terms may be unknown

---

### POS Tagging - `lexprep en pos`

**Library**: [spaCy](https://spacy.io/) by Explosion AI

**Model**: `en_core_web_sm` (small English model)

**How it works**:
1. Tokenizes input text
2. Uses a statistical model trained on [OntoNotes 5](https://catalog.ldc.upenn.edu/LDC2013T19) corpus
3. Assigns Universal POS tags (UPOS) and fine-grained Penn Treebank tags

**Output**:
- `pos`: Universal POS tag (NOUN, VERB, ADJ, etc.)
- `tag`: Fine-grained tag (NN, VBZ, JJ, etc.)
- `lemma`: Base form of the word

**Tag schemes**:
- UPOS: 17 universal categories ([Universal Dependencies](https://universaldependencies.org/u/pos/))
- Penn Treebank: ~45 fine-grained tags

**Accuracy**: spaCy's `en_core_web_sm` reports ~97% POS accuracy on standard benchmarks.

**Reference**: https://spacy.io/models/en

---

## Persian Tools

### G2P (Grapheme-to-Phoneme) - `lexprep fa g2p`

**Library**: [PersianG2p](https://github.com/PasaOpasen/PersianG2P) by Demetry Pascal

**How it works**:
1. Normalizes Persian text (Unicode normalization)
2. Uses a neural model trained on Persian pronunciation data
3. Outputs romanized pronunciation with standard vowel markers

**Output format**: Romanized Persian (e.g., "salaam" for hello)

**Note**: This is a deep learning-based approach; accuracy depends on the training data coverage.

---

### Syllable Counting - `lexprep fa syllables`

**Methods available**:

#### 1. Orthographic (`--method orthographic`)
- **Input**: Persian text in Arabic script
- **Algorithm**:
  1. Identifies Persian vowel letters
  2. Segments word at vowel boundaries
  3. Handles compound vowels
- **Accuracy**: Heuristic-based, ~85-90% estimated (no formal benchmark)

#### 2. Phonetic (`--method phonetic` or `--with-g2p`)
- **Input**: Romanized pronunciation string
- **Algorithm**:
  1. Scans for vowel patterns (long and short vowels)
  2. Uses longest-first matching to avoid double-counting
  3. Each vowel pattern = one syllable
- **Accuracy**: More accurate than orthographic when G2P output is correct

**Limitations**:
- Loanwords may not follow standard patterns
- Words without diacritics may be ambiguous
- Compound words may have unclear boundaries

---

### POS Tagging - `lexprep fa pos`

**Library**: [Stanza](https://stanfordnlp.github.io/stanza/) by Stanford NLP Group

**How it works**:
1. Uses neural models trained on UD Persian Seraji treebank
2. Assigns Universal POS tags (UPOS)
3. Provides lemmatization

**Output**:
- `pos`: Universal POS tag (NOUN, VERB, ADJ, etc.)
- `lemma`: Base form of the word

**Accuracy**: 97.4% UPOS accuracy (official Stanza benchmark)

**Reference**:
```bibtex
@inproceedings{qi2020stanza,
  title={Stanza: A Python Natural Language Processing Toolkit for Many Human Languages},
  author={Qi, Peng and Zhang, Yuhao and Zhang, Yuhui and Bolton, Jason and Manning, Christopher D.},
  booktitle={ACL 2020 System Demonstrations},
  year={2020}
}
```

---

## Japanese Tools

### POS Tagging - `lexprep ja pos`

**Methods available**:

#### 1. UniDic (`--method unidic`)
**Libraries**: [Fugashi](https://github.com/polm/fugashi) + [UniDic](https://clrd.ninjal.ac.jp/unidic/)

**How it works**:
1. Uses MeCab morphological analyzer with UniDic dictionary
2. Provides detailed Japanese POS hierarchy (pos1, pos2, pos3, pos4)
3. Extracts lemma (dictionary form)
4. Maps `pos1` to English gloss via `pos_map`

**Output**:
- `pos1..pos4`: Japanese POS hierarchy
- `pos_en`: English gloss (noun, verb, adjective)
- `lemma`: Dictionary form

**Accuracy**: UniDic is the standard dictionary for modern Japanese NLP, maintained by NINJAL.

#### 2. Stanza (`--method stanza`)
**Library**: [Stanza](https://stanfordnlp.github.io/stanza/) by Stanford NLP

**How it works**:
1. Uses neural models trained on Universal Dependencies treebanks
2. Provides Universal POS tags (UPOS)
3. Cross-linguistically consistent annotations

**Output**:
- `pos`: Universal POS tag (NOUN, VERB, ADJ, etc.)

**Accuracy**: Stanza Japanese models achieve ~97% UPOS accuracy on UD Japanese treebanks.

**Reference**:
```bibtex
@inproceedings{qi2020stanza,
  title={Stanza: A Python Natural Language Processing Toolkit for Many Human Languages},
  author={Qi, Peng and Zhang, Yuhao and Zhang, Yuhui and Bolton, Jason and Manning, Christopher D.},
  booktitle={ACL 2020 System Demonstrations},
  year={2020}
}
```

---

## Sampling Tools

### Stratified Sampling - `lexprep sample stratified`

**Algorithm**:
1. Divides data into N quantile bins based on score column using pandas `qcut`
2. Allocates sample quota per bin (default: equal; optional: proportional, optimal, fixed)
3. Samples rows from each bin with a fixed random seed (default: `--seed 19`)
4. Adds `bin_id` column to all output rows

**Output ZIP contents** (`{name}__stratified_sampling__all__{ts}.zip`):
- `{name}__sample.xlsx` — selected rows with `bin_id`
- `{name}__excluded.xlsx` — non-selected rows with `bin_id`
- `sampling_audit.xlsx` — per-bin table: population, selected, excluded, allocation method
- `run_manifest.json` — parameters for exact reproduction

**Use case**: Creating balanced experimental stimuli across frequency ranges.

---

### Multi-file Shuffle - `lexprep sample shuffle`

**Algorithm**:
1. Verifies all files have equal row counts
2. Generates a single random permutation of row indices (fixed `--seed`)
3. Applies the same permutation to all files simultaneously
4. Maintains row correspondence across files

**Output ZIP contents** (`shuffle__row_shuffle__all__{ts}.zip`):
- `{name}_shuffled.{ext}` for each input file
- `run_manifest.json` — seed and parameters

**Use case**: Randomizing parallel data (e.g., words + translations + audio filenames).

---

## Universal Tools

### Word Length - `lexprep length`

**Method**: Unicode codepoint count (`len()` in Python 3).

**Algorithm**:
1. Treats each word as a Python string
2. Returns `len(word)` — the number of Unicode codepoints
3. Language-agnostic: works correctly for Arabic-script, CJK, and Latin words

**Output column**: `length_chars`
**Method tag in manifest**: `"unicode_codepoints"`

**Note**: Codepoint count differs from byte count and grapheme count. For most wordlist
use cases (frequency banding, stimulus matching) codepoint count is the standard measure.

---

## Reproducibility Pack

Every lexprep command writes a **ZIP file** instead of a plain spreadsheet. The ZIP contains:

| File | Contents |
|------|----------|
| `run_manifest.json` | Tool name, version, timestamp (UTC), input filename, added columns, library versions, seed |
| `{name}__enriched.{ext}` | Original data + added columns (language tools) |
| `{name}__sample.{ext}` | Selected rows (stratified sampling only) |
| `{name}__excluded.{ext}` | Non-selected rows (stratified sampling only) |
| `sampling_audit.xlsx` | Per-bin statistics (stratified sampling only) |
| `{name}_shuffled.{ext}` | Shuffled files (shuffle only, one per input) |

### ZIP filename format

```
{input_basename}__{tool}__{language}__{YYYYMMDDTHHMMSSZ}.zip
```

Example: `wordlist__g2p__fa__20260220T143000Z.zip`

### run_manifest.json example

```json
{
  "lexprep_version": "1.0.0",
  "timestamp_utc": "2026-02-20T14:30:00+00:00",
  "tool": "g2p",
  "language": "fa",
  "citation": { "doi": "10.5281/zenodo.18713755", "...": "..." },
  "input": {
    "original_filename": "wordlist.xlsx",
    "file_type": "xlsx",
    "row_count": 500,
    "column_mapping": { "word_column": "word" }
  },
  "pipeline": {
    "added_columns": ["pronunciation", "g2p_error"],
    "libraries": [{ "name": "PersianG2p", "version": "0.1.5" }]
  }
}
```

For sampling, a `reproducibility` block is added with `seed` and `parameters`, and a `sampling` block with per-bin allocation details.

---

**Note**: Accuracy figures are estimates based on published benchmarks. Actual accuracy may vary depending on your specific data domain.
