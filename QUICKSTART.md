# Quick Start Guide

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate  # Linux/Mac
pip install .
```

## Basic Commands

### Persian

```bash
# G2P transcription
lexprep fa g2p input.xlsx output.xlsx -c word

# Syllable counting (orthographic)
lexprep fa syllables input.xlsx output.xlsx -c word

# G2P + Syllables in one step
lexprep fa syllables input.xlsx output.xlsx -c word --with-g2p

# POS tagging
lexprep fa pos input.xlsx output.xlsx -c word -m /path/to/model
```

### English

```bash
# G2P (ARPAbet)
lexprep en g2p input.xlsx output.xlsx -c word

# Syllables
lexprep en syllables input.xlsx output.xlsx -c word

# POS tagging
lexprep en pos input.xlsx output.xlsx -c word
```

### Japanese

```bash
# POS with UniDic (detailed)
lexprep ja pos input.xlsx output.xlsx -c word --method unidic

# POS with Stanza (universal)
lexprep ja pos input.xlsx output.xlsx -c word --method stanza
```

### Sampling

```bash
# Stratified sampling (outputs ZIP with sample + excluded + audit)
lexprep sample stratified input.xlsx output.xlsx --score frequency --n 100 --bins 3 --seed 19

# Shuffle multiple files
lexprep sample shuffle file1.xlsx file2.xlsx output_dir/
```

### Length (language-agnostic)

```bash
# Add length_chars column (Unicode codepoint count)
lexprep length words.xlsx output.xlsx -c word
```


## Output

All commands output a **ZIP reproducibility pack** containing the enriched data plus a `run_manifest.json` with tool version, parameters, and library versions.

See [METHODS.md](METHODS.md#reproducibility-pack) for details.


## File Formats

- **TXT**: One word per line
- **CSV/TSV**: Tables with headers
- **XLSX**: Excel files

Use `-c` / `--col` to specify word column.


## More Info

See [README.md](README.md) for full documentation.
