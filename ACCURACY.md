# Evaluation & Accuracy

This document provides accuracy information for each tool used in lexprep. Where official benchmarks exist, they are cited with references. Where no formal evaluation exists, estimates are clearly marked.

---

## Summary Table

| Tool | Library | Accuracy | Source | Notes |
|------|---------|----------|--------|-------|
| **English G2P** | g2p-en | High | Park & Kim (2019) | CMU Dict (~134,500 words) |
| **English Syllables** | pyphen | No benchmark | Pyphen | TeX-compatible hyphenation |
| **English POS** | spaCy | 96-97% | Explosion AI | Trained on OntoNotes 5 |
| **Persian G2P** | PersianG2p | No formal benchmark | - | ~48,000 word dictionary + neural |
| **Persian Syllables** | Heuristic | **Estimate**: ~85-95% | - | No formal evaluation |
| **Persian POS** | Stanza | **97.69%** | Qi et al. (2020) | UD Persian Seraji treebank |
| **Japanese POS (Stanza)** | Stanza | **96-97%** | Qi et al. (2020) | UD Japanese GSD treebank |
| **Japanese POS (UniDic)** | Fugashi+UniDic | Widely used | NINJAL | MeCab default for Japanese |

---

## English

### G2P (g2p-en)

**Library**: [g2p-en](https://github.com/Kyubyong/g2p) by Park & Kim (2019)

**How it works**:
1. Spells out numbers and currency symbols
2. Disambiguates homographs using POS tags
3. Looks up CMU Pronouncing Dictionary (~134,500 words)
4. Predicts OOV words using neural seq2seq model

**Accuracy**:
- Dictionary words: 100% (by definition - dictionary lookup)
- OOV words: Neural prediction (no published benchmark)

**Reference**:
- GitHub: https://github.com/Kyubyong/g2p
- CMU Dictionary: http://www.speech.cs.cmu.edu/cgi-bin/cmudict

**License**: Apache License 2.0

---

### Syllables (pyphen)

**Library**: [pyphen](https://github.com/Kozea/Pyphen)

**How it works**:
- Uses TeX-compatible hyphenation dictionaries from LibreOffice (Hunspell format)
- Hyphenates words using language-specific patterns
- Syllable count is inferred from hyphenation segments

**Accuracy**:
- No formal benchmark published

**Reference**:
- GitHub: https://github.com/Kozea/Pyphen
- Patterns: http://www.hyphenation.org

**License**: LGPL/GPL/MPL (tri-license)

---

### POS Tagging (spaCy)

**Library**: [spaCy](https://spacy.io/) by Explosion AI

**Model**: `en_core_web_sm` (trained on OntoNotes 5)

**Official Benchmarks** (from spaCy model page):

| Metric | Score |
|--------|-------|
| Token accuracy | 97% |
| POS accuracy | 97% |
| Sentence segmentation F1 | 91% |

**Reference**:
- spaCy models: https://spacy.io/models/en
- Honnibal & Montani (2017): spaCy 2: Natural language understanding with Bloom embeddings

**License**: MIT

---

## Persian

### G2P (PersianG2p)

**Library**: [PersianG2p](https://github.com/PasaOpasen/PersianG2P) by Demetry Pascal (forked from AzamRabiee/Persian_G2P)
**What lexprep imports**: PyPI package `PersianG2p` (Home-page: https://github.com/PasaOpasen/PersianG2P)

**How it works**:
1. Normalizes text
2. Looks up dictionary (~48,000 words with `use_large=True`)
3. Predicts unknown words using neural network

**Accuracy**:
- **No formal benchmark published**
- Dictionary includes ~48,000 Persian words
- Neural model handles OOV words

**Reference**:
- GitHub: https://github.com/PasaOpasen/PersianG2P
- Original: https://github.com/AzamRabiee/Persian_G2P

**License**: MIT

---

### Syllables (Heuristic)

**Method**: Custom heuristic in lexprep

**How it works**:
- **Orthographic**: Counts vowel patterns in Persian spelling
- **Phonetic**: Counts vowels in G2P output

**Accuracy**:
- **No formal benchmark** - these are estimates:
  - Orthographic: ~65-80%
  - Phonetic: ~90-95% (depends on G2P accuracy)

**Limitations**:
- Heuristic-based, not ML-based
- May fail on loanwords, compound words

---

### POS Tagging (Stanza)

**Library**: [Stanza](https://stanfordnlp.github.io/stanza/) by Stanford NLP Group

**Model**: Persian Seraji (UD Persian Seraji treebank)

**Official Benchmarks** (Stanza v1.5.1 on UD v2.12):

| Dataset | Tokens | UPOS | Lemma |
|---------|--------|------|-------|
| Persian Seraji | 100% | **97.69%** | 98.18% |

**Output**: Universal POS tags (UPOS) + Lemma
**Treebank**: UD Persian Seraji

**Reference**:
```bibtex
@inproceedings{qi2020stanza,
    title={Stanza: A Python Natural Language Processing Toolkit for Many Human Languages},
    author={Qi, Peng and Zhang, Yuhao and Zhang, Yuhui and Bolton, Jason and Manning, Christopher D.},
    booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations},
    year={2020}
}
```

**Official Sources**:
- Performance: https://stanfordnlp.github.io/stanza/performance.html
- Paper: https://aclanthology.org/2020.acl-demos.14/

**License**: Apache 2.0

---

## Japanese

### POS Tagging (Stanza)

**Library**: [Stanza](https://stanfordnlp.github.io/stanza/) by Stanford NLP Group

**Official Benchmarks** (Stanza v1.5.1 on UD v2.12):

| Dataset | Tokens | UPOS | Lemma |
|---------|--------|------|-------|
| Japanese GSD | 97.37% | **96.38%** | 96.02% |
| Japanese GSDLUW | 96.32% | **95.10%** | 94.67% |

**Output**: Universal POS tags (17 categories)
**Best for**: Cross-lingual research, UD compatibility

**Reference**:
```bibtex
@inproceedings{qi2020stanza,
    title={Stanza: A Python Natural Language Processing Toolkit for Many Human Languages},
    author={Qi, Peng and Zhang, Yuhao and Zhang, Yuhui and Bolton, Jason and Manning, Christopher D.},
    booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations},
    year={2020}
}
```

**Official Sources**:
- Performance: https://stanfordnlp.github.io/stanza/performance.html
- Paper: https://aclanthology.org/2020.acl-demos.14/

**License**: Apache 2.0

---

### POS Tagging (UniDic)

**Libraries**:
- [Fugashi](https://github.com/polm/fugashi) - Python wrapper for MeCab
- [UniDic](https://clrd.ninjal.ac.jp/unidic/) - Dictionary by NINJAL

**How it works**:
- MeCab morphological analyzer with UniDic dictionary
- UniDic contains ~240,000+ entries
- Standard for Japanese NLP since 2001
- Maps `pos1` to English gloss (`pos_en`) using lexprep's `pos_map`

**Accuracy**:
- Widely used - MeCab+UniDic is the **de facto standard** for Japanese morphological analysis

**Output**: Detailed Japanese POS tags
**Best for**: Japanese linguistics, detailed morphological analysis

**Reference**:
- UniDic: https://clrd.ninjal.ac.jp/unidic/
- Fugashi: https://github.com/polm/fugashi
- MeCab: https://taku910.github.io/mecab/

**License**:
- Fugashi: MIT
- UniDic: BSD/GPL/LGPL (varies by version)

---

## Licensing & Usage

All libraries used by lexprep are open-source with permissive licenses:

| Library | License | Can Use? |
|---------|---------|----------|
| g2p-en | MIT | Yes |
| pyphen | LGPL/GPL/MPL | Yes |
| spaCy | MIT | Yes |
| PersianG2p | MIT | Yes |
| Stanza | Apache 2.0 | Yes |
| Fugashi | MIT | Yes |
| UniDic | BSD/GPL/LGPL | Yes |

**lexprep uses these libraries as dependencies** (via `pip install`) without modification. This is standard practice and does not require forking.

---

## Recommendations for Researchers

1. **Report tool versions** in your papers (e.g., "spaCy v3.7", "Stanza v1.5.1")
2. **Validate on your domain** - accuracy varies by text type
3. **Cite underlying tools** - see each library's citation format
4. **For Persian syllables**: manually verify a sample since no formal benchmark exists

---

## References

- Park, K., & Kim, J. (2019). g2p-en. https://github.com/Kyubyong/g2p
- Explosion AI. spaCy: Industrial-strength NLP. https://spacy.io/
- Qi, P., et al. (2020). Stanza: A Python NLP Toolkit. ACL 2020.
- NINJAL. UniDic. https://clrd.ninjal.ac.jp/unidic/
