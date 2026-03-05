---
title: "LexPrep: An Open-Source Toolkit for Wordlist Processing and Stimulus Preparation"
tags:
  - Python
  - psycholinguistics
  - wordlists
  - stimulus preparation
  - grapheme-to-phoneme
  - syllable counting
  - POS tagging
  - stratified sampling
  - reproducibility
authors:
  - name: Sajjad Mazaherizaveh
    orcid: 0009-0001-0465-0444
    affiliation: 1
affiliations:
  - name: Independent Researcher, Rome, Italy
    index: 1
date: 2026
bibliography: paper.bib
---

# Summary

LexPrep is an open-source Python toolkit for processing wordlists and preparing lexical stimuli for experimental research. It provides grapheme-to-phoneme (G2P) transcription, syllable counting, part-of-speech (POS) tagging, stratified sampling, and synchronized multi-file shuffling for wordlists in Persian, English, and Japanese. LexPrep is available as a command-line tool (installed via `pip install .` after cloning the repository) and as a browser-based web interface at [lexprep.net](https://lexprep.net), allowing researchers to process wordlists without writing code or managing Python dependencies. Source code is hosted at [github.com/sajjad-mazaheri/lexprep](https://github.com/sajjad-mazaheri/lexprep) under the MIT license.

Unlike general-purpose NLP libraries such as spaCy [@honnibal2020spacy] and Stanza [@qi2020stanza], which are primarily designed for processing running text through
document-oriented pipelines, LexPrep operates on isolated words in tabular formats (CSV, XLSX, TSV, TXT). It reads a researcher's data file, applies the requested processing to a specified word column, appends results as new columns, and writes the enriched file back, preserving the original data structure throughout. Every run produces a machine-readable `run_manifest.json` that records the software version, parameters, input file hash, and timestamp, making each processing step independently reproducible.

# Statement of Need

Researchers in psycholinguistics, reading science, and cognitive neuroscience routinely prepare controlled wordlists as experimental stimuli. A typical workflow involves obtaining phonetic transcriptions, counting syllables, assigning part-of-speech tags, and then drawing a balanced subset from the full list based on a variable such as lexical frequency. These steps currently require combining several separate tools, often through manual scripting to bridge incompatible interfaces and output formats. The resulting process is difficult to document and rarely shared in full, creating a gap in reproducibility: published studies describe their stimulus selection criteria, but the specific steps that produced the final wordlist are seldom available for inspection or replication.

For languages such as Persian, the situation is more challenging. There is no integrated toolkit for Persian wordlist annotation. Researchers must locate and configure individual libraries for G2P, POS tagging, and syllable estimation, each with its own input conventions. Japanese poses similar challenges when researchers need both Universal Dependencies tags for cross-linguistic comparison and detailed morphological analysis via UniDic.

LexPrep addresses this gap by providing a single interface that handles the full annotation-to-sampling pipeline. A researcher can upload a spreadsheet of words, apply G2P transcription, syllable counting, and POS tagging in sequence, then draw a stratified sample, and download the result as a ready-to-use file. Every step operates on the same tabular structure, eliminating the need for format conversion or custom glue code. The web interface at lexprep.net extends this accessibility to researchers who do not program, requiring no software installation.

# State of the Field

Several tools address aspects of lexical stimulus preparation, but none provide an integrated wordlist-processing pipeline across multiple languages.

**General NLP toolkits.** spaCy [@honnibal2020spacy] and Stanza [@qi2020stanza] offer tokenization, POS tagging, lemmatization, and dependency parsing for many languages. However, they are designed for sentence- and document-level analysis. Using them for wordlist processing requires writing wrapper scripts to handle file I/O, iterate over isolated words, and reformat outputs into tabular form, which are exactly the steps that LexPrep automates.

**Language-specific G2P tools.** For English, g2p-en [@park2019g2p] provides phonemic transcription using the CMU Pronouncing Dictionary and a neural fallback model. For Persian, PersianG2p [@pascal2020persiang2p] offers dictionary lookup and neural prediction. LexPrep integrates both, handling input normalization and output formatting so that researchers interact with a single interface regardless of language.

**Psycholinguistic databases and generators.** Tools such as Wuggy [@keuleers2010wuggy] generate pseudowords, and databases such as SUBTLEX [@brysbaert2009subtlex] provide frequency norms. These are complementary to LexPrep: a researcher might draw frequency values from SUBTLEX, add them to a wordlist, and then use LexPrep's stratified sampling to select a balanced subset.

**Syllable counting.** For English, pyphen [@pyphen] provides TeX-based hyphenation that can approximate syllable counts. For Persian, no widely adopted tool exists; LexPrep provides both orthographic (heuristic) and phonetic (G2P-based) syllable counting methods, with documented accuracy estimates (see [ACCURACY.md](ACCURACY.md)).

LexPrep's contribution is not in the underlying NLP models but in the integration layer: a consistent interface that takes a wordlist file as input, applies multiple annotation and sampling steps, and returns an enriched file with a full audit trail, requiring no programming from the end user.

# Software Design

LexPrep is structured as a Python package with three layers.

**Core processing modules.** Each supported language (Persian, English, Japanese) has a dedicated module wrapping the relevant NLP libraries. The English module integrates g2p-en for phonemic transcription, pyphen for syllable counting, and spaCy for POS tagging. The Persian module integrates PersianG2p and Stanza, and includes a custom heuristic syllable counter. The Japanese module supports both Stanza for Universal POS tags and Fugashi [@fugashi] with UniDic [@unidic] for detailed morphological analysis, with an automatic mapping from Japanese POS categories to English glosses.

**CLI interface.** All functionality is exposed through a command-line interface organized by language and tool (e.g., `lexprep fa g2p`, `lexprep en syllables`, `lexprep sample stratified`). The CLI reads tabular files, applies the requested processing, and writes output with new columns appended, preserving the researcher's original data. Multiple steps can be chained in sequence.

**Web interface.** A Flask-based web application provides browser-based access to all processing tools. The backend uses asynchronous job processing for large files and a model caching layer that reduces repeated processing time by approximately three orders of magnitude. LexPrep is built around a core library that the web interface exposes; all processing logic resides in importable Python modules that can be used independently of the web layer.

**Sampling module.** The stratified sampling tool divides data into quantile-based bins on a specified score column and draws proportional or equal samples from each bin. The multi-file shuffle tool generates a single random permutation and applies it to multiple files simultaneously, maintaining row correspondence across parallel stimulus sets.

**Reproducibility.** Every processing run generates a `run_manifest.json` file recording the LexPrep version, the operation performed, all parameters, input file hash, and a timestamp. This manifest allows any collaborator or reviewer to verify or reproduce the exact processing that was applied.

**Language dependencies are optional.** The base installation (`pip install .`) provides only the core; language-specific dependencies are installed via extras (`pip install ".[fa]"`, `".[en]"`, `".[ja]"`), keeping the base installation lightweight.

**Testing and CI/CD.** The repository includes automated unit and integration tests covering the command-line interface and the web application. Continuous integration pipelines via GitHub Actions run tests on every pull request and merge to the main branch, for both the development and deployment environments.

# Research Impact Statement

LexPrep is currently used in ongoing reading research at Sapienza University of Rome in collaboration with New York University, specifically for preparing Persian wordlist stimuli for studies on crowding effects in reading. The web interface at lexprep.net provides a low-barrier entry point for researchers across the language sciences, and the software has been adopted by researchers working in linguistics and psycholinguistics.

# AI Usage Disclosure

Generative AI tools (OpenAI ChatGPT-4o and Anthropic Claude Sonnet/Opus) were used as coding assistants during development. Their use included: suggesting code snippets during implementation, generating initial drafts of documentation text, and assisting with debugging. The author reviewed, tested, and edited all AI-generated outputs before inclusion. All scientific design decisions, the software architecture, the choice of integrated libraries, the validation methodology, and the research framing in this paper were made by the author. No AI tool was used in the peer review process.

# Acknowledgements

Development of LexPrep was motivated by research conducted during the author's M.Sc. studies at Sapienza University of Rome, in collaboration with Prof. Marialuisa Martelli (Sapienza) and Prof. Denis G. Pelli (New York University). The author thanks the developers of the open-source libraries that LexPrep integrates: g2p-en, pyphen, spaCy, PersianG2p, Stanza, Fugashi, and UniDic.

# References
