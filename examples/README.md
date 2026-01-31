# Examples

This folder contains sample input files to demonstrate lexprep functionality.

## Files

- `sample_persian.txt` - Simple Persian words (one per line) for G2P and syllabification demos
- `sample_words.csv` - English word list with frequencies for stratified sampling demo

## Quick Examples

### Persian G2P (Grapheme-to-Phoneme)
```bash
lexprep fa g2p examples/sample_persian.txt output_g2p.xlsx
```

### Stratified Sampling
```bash
lexprep sample stratified examples/sample_words.csv output_sample.xlsx --score-col frequency --n-total 5 --bins 3
```

See the main [README.md](../README.md) for comprehensive usage instructions.
