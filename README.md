<p align="center">
  <h1 align="center">lexprep</h1>
  <p align="center">
    <strong>Linguistic Data Preparation Toolkit for Wordlists</strong>
  </p>
  <p align="center">
    G2P - Syllables - POS Tagging - Sampling<br>
    Persian - English - Japanese
  </p>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
</p>

---

## What is lexprep?

**lexprep** is a  toolkit designed for **linguistic data preparation**. It processes **wordlists**  and provides:

- **G2P Transcription**: Convert words to phonetic representations
- **Syllable Counting**: Count syllables using orthographic or phonetic methods
- **POS Tagging**: Assign part-of-speech tags to words
- **Sampling**: Stratified sampling and multi-file shuffling

### Key Difference from Other NLP Tools

Most NLP libraries (spaCy, Stanza, etc.) are designed for **text processing** - analyzing sentences and documents.

**lexprep** is designed for **wordlist processing** - working with isolated words in spreadsheets:

| Tool | Input | Use Case |
|------|-------|----------|
| spaCy, Stanza | Running text/sentences | Document analysis, NER, parsing |
| **lexprep** | Wordlists (CSV/XLSX/TXT) | stimulus preparation |


## Supported Languages & Tools

| Language | G2P | Syllables | POS |
|----------|-----|-----------|-----|
| Persian | PersianG2p | Heuristic | Stanza |
| English | g2p-en | pyphen | spaCy |
| Japanese | - | - | Stanza / UniDic |

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
# Stratified sampling
lexprep sample stratified data.xlsx output.xlsx --score-col frequency --n-total 100 --bins 3

# Shuffle multiple files
lexprep sample shuffle-rows file1.xlsx file2.xlsx output_dir/
```

---

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
- Download results automatically
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
