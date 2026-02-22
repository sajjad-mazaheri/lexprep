<p align="center">
  <h1 align="center">lexprep</h1>
  <p align="center">
    <strong>Linguistic Data Preparation Toolkit for Wordlists</strong>
  </p>
  <p align="center">
    G2P - Syllables - POS Tagging - Length - Sampling<br>
    Persian - English - Japanese
  </p>
</p>

<p align="center">
  <a href="https://doi.org/10.5281/zenodo.18713755">
    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.18713755.svg" alt="DOI">
  </a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
</p>

---

## What is lexprep?

**lexprep** is a  toolkit designed for **linguistic data preparation**. It processes **wordlists**  and provides:

- **G2P Transcription**: Convert words to phonetic representations
- **Syllable Counting**: Count syllables using orthographic or phonetic methods
- **POS Tagging**: Assign part-of-speech tags to words
- **Length**: Count Unicode codepoints per word (`length_chars`)
- **Sampling**: Stratified sampling and multi-file shuffling
- **Reproducibility Pack**: Every command outputs a ZIP with data + `run_manifest.json`

### Key Difference from Other NLP Tools

Most NLP libraries (spaCy, Stanza, etc.) are designed for **text processing** - analyzing sentences and documents.

**lexprep** is designed for **wordlist processing** - working with isolated words in spreadsheets:

| Tool | Input | Use Case |
|------|-------|----------|
| spaCy, Stanza | Running text/sentences | Document analysis, NER, parsing |
| **lexprep** | Wordlists (CSV/XLSX/TXT) | stimulus preparation |


## Supported Languages & Tools

| Language | G2P | Syllables | POS | Length |
|----------|-----|-----------|-----|--------|
| Persian | PersianG2p | Heuristic | Stanza | ✓ |
| English | g2p-en | pyphen | spaCy | ✓ |
| Japanese | - | - | Stanza / UniDic | ✓ |
| (any) | - | - | - | ✓ |

---

## Installation

```bash
pip install .              # Core only
pip install ".[fa]"        # + Persian
pip install ".[en]"        # + English
pip install ".[ja]"        # + Japanese
pip install ".[fa,en,ja]"  # All languages
```

---

## Quick Start

### Persian
```bash
# G2P transcription
lexprep fa g2p words.xlsx output.xlsx -c word

# Syllable counting (with automatic G2P)
lexprep fa syllables words.xlsx output.xlsx -c word --with-g2p

# POS tagging
lexprep fa pos words.xlsx output.xlsx -c word
```

### English
```bash
# G2P (ARPAbet phonemes)
lexprep en g2p words.xlsx output.xlsx -c word

# Syllable counting
lexprep en syllables words.xlsx output.xlsx -c word

# POS tagging
lexprep en pos words.xlsx output.xlsx -c word
```

### Japanese
```bash
# POS with UniDic (detailed tags)
lexprep ja pos words.xlsx output.xlsx -c word --method unidic

# POS with Stanza (universal tags)
lexprep ja pos words.xlsx output.xlsx -c word --method stanza
```

**Which Japanese method should I use?**
- Use **Stanza** when you want **Universal POS (English tags)** for cross-lingual comparison or UD-style annotation.
- Use **UniDic** when you want **detailed Japanese tags** (pos1..pos4) for linguistic analysis. lexprep also returns `pos_en` (English gloss) for UniDic output automatically via `pos_map`.

### Sampling
```bash
# Stratified sampling — outputs ZIP with sample + excluded + audit
lexprep sample stratified data.xlsx output.xlsx --score frequency --n 100 --bins 3 --seed 19

# Shuffle multiple files (synchronized row permutation)
lexprep sample shuffle file1.xlsx file2.xlsx output_dir/
```

### Length (language-agnostic)
```bash
# Add length_chars column (Unicode codepoint count)
lexprep length words.xlsx output.xlsx -c word
```

---

## Output Format

Every command produces a **ZIP reproducibility pack** instead of a plain file:

```
words__g2p__fa__20260220T143000Z.zip
├── run_manifest.json          ← tool, version, parameters, libraries
└── words__enriched.xlsx       ← original data + added columns
```

For sampling:
```
data__stratified_sampling__all__20260220T143000Z.zip
├── run_manifest.json
├── data__sample.xlsx          ← selected rows with bin_id
├── data__excluded.xlsx        ← non-selected rows
└── sampling_audit.xlsx        ← per-bin statistics
```

The `run_manifest.json` records the exact tool version, timestamp, seed, and library versions used — enabling exact reproduction of any processing step.

## File Formats

| Format | Support |
|--------|---------|
| Excel (.xlsx) | Supported |
| CSV (.csv) | Supported |
| TSV (.tsv) | Supported |
| Plain text (.txt) | Supported (one word per line) |

Use `-c` or `--column-name` to specify the word column.

---

## Web Interface

lexprep includes a **web-based UI** for processing wordlists through your browser:


Features:
- Upload files (Excel, CSV, TSV, TXT)
- Select language and tool
- Process wordlists with all available tools
- Download results as a ZIP reproducibility pack
- Fast processing with model caching

---

## Documentation

- [METHODS.md](METHODS.md) - How each algorithm works
- [ACCURACY.md](ACCURACY.md) - Benchmark results and evaluation
- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines

---

## Notes

- English POS requires: `python -m spacy download en_core_web_sm`
- Persian and Japanese Stanza models download automatically on first use
- Persian syllables use heuristic methods (see [ACCURACY.md](ACCURACY.md))

---

## Underlying Libraries

lexprep integrates these excellent open-source libraries:

### English
- **[g2p-en](https://github.com/Kyubyong/g2p)** - G2P using CMU Dictionary + neural model (Park & Kim, 2019) - MIT License
- **[pyphen](https://github.com/Kozea/Pyphen)** - TeX hyphenation patterns - LGPL/GPL/MPL
- **[spaCy](https://spacy.io/)** - Industrial-strength NLP (Explosion AI) - MIT License

### Persian
- **[PersianG2p](https://github.com/PasaOpasen/PersianG2P)** - Persian G2P (Demetry Pascal, forked from AzamRabiee) - MIT License
- **[Stanza](https://stanfordnlp.github.io/stanza/)** - Stanford NLP toolkit for POS tagging (Qi et al., 2020) - Apache 2.0

### Japanese
- **[Stanza](https://stanfordnlp.github.io/stanza/)** - Stanford NLP toolkit (Qi et al., 2020) - Apache 2.0
- **[Fugashi](https://github.com/polm/fugashi)** - MeCab wrapper - MIT License
- **[UniDic](https://clrd.ninjal.ac.jp/unidic/)** - Japanese dictionary (NINJAL) - GPL/LGPL/BSD

All libraries are used as dependencies without modification. See their respective licenses for details.

---

## License

MIT License - See [LICENSE](LICENSE) for details.
