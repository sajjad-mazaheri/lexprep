from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer
from rich import print

from .io.files import read_table, read_wordlist, write_table

app = typer.Typer(
    add_completion=False,
    help="lexprep - wordlist preparation toolkit",
    no_args_is_help=True
)

# Sub-commands for each language
fa_app = typer.Typer(help="Persian tools", no_args_is_help=True)
en_app = typer.Typer(help="English tools", no_args_is_help=True)
ja_app = typer.Typer(help="Japanese tools", no_args_is_help=True)
sample_app = typer.Typer(help="Sampling tools", no_args_is_help=True)

app.add_typer(fa_app, name="fa")
app.add_typer(en_app, name="en")
app.add_typer(ja_app, name="ja")
app.add_typer(sample_app, name="sample")


# Persian Commands


@fa_app.command("g2p")
def fa_g2p(
    input_file: str = typer.Argument(..., help="Input file"),
    output_file: str = typer.Argument(..., help="Output file"),
    word_col: Optional[str] = typer.Option(None, "--col", "-c", help="Word column name"),
):
    from .fa.g2p import transcribe_words

    df, col = read_wordlist(input_file, word_col=word_col)
    words = df[col].astype(str).fillna("").tolist()
    results = transcribe_words(words)

    pron, err = [], []
    idx = 0
    for w in words:
        if not str(w).strip():
            pron.append("")
            err.append("")
        else:
            r = results[idx]
            pron.append(r.pronunciation)
            err.append(r.error or "")
            idx += 1

    df["pronunciation"] = pron
    df["g2p_error"] = err
    write_table(df, output_file)
    print(f"[green]✓[/green] Saved: {output_file}")


@fa_app.command("syllables")
def fa_syllables(
    input_file: str = typer.Argument(..., help="Input file"),
    output_file: str = typer.Argument(..., help="Output file"),
    word_col: Optional[str] = typer.Option(None, "--col", "-c", help="Word column name"),
    method: str = typer.Option(
        "orthographic", "--method", "-m", help="Method: orthographic or phonetic"
    ),
    pron_col: str = typer.Option(
        "pronunciation", "--pron-col", help="Pronunciation column (phonetic)"
    ),
    with_g2p: bool = typer.Option(
        False, "--with-g2p", help="Run G2P first, then count syllables (phonetic)"
    ),
):
    """Count syllables in Persian words."""
    from .fa.syllables import count_syllables_from_pronunciation, syllabify_orthographic

    if with_g2p:
        from .fa.g2p import transcribe_words
        df, col = read_wordlist(input_file, word_col=word_col)
        words = df[col].astype(str).fillna("").tolist()
        results = transcribe_words(words)

        pron, count = [], []
        idx = 0
        for w in words:
            if not str(w).strip():
                pron.append("")
                count.append(0)
            else:
                r = results[idx]
                pron.append(r.pronunciation)
                count.append(count_syllables_from_pronunciation(r.pronunciation))
                idx += 1

        df["pronunciation"] = pron
        df["syllables"] = count
    elif method == "phonetic":
        df = read_table(input_file)
        if pron_col not in df.columns:
            raise typer.BadParameter(f"Column '{pron_col}' not found")
        df["syllables"] = df[pron_col].astype(str).apply(count_syllables_from_pronunciation)
    else:
        df, col = read_wordlist(input_file, word_col=word_col)
        sylls, counts = [], []
        for w in df[col].astype(str).tolist():
            s, c = syllabify_orthographic(w)
            sylls.append(s)
            counts.append(c)
        df["syllabified"] = sylls
        df["syllables"] = counts

    write_table(df, output_file)
    print(f"[green]✓[/green] Saved: {output_file}")


@fa_app.command("pos")
def fa_pos(
    input_file: str = typer.Argument(..., help="Input file"),
    output_file: str = typer.Argument(..., help="Output file"),
    word_col: Optional[str] = typer.Option(None, "--col", "-c", help="Word column name"),
):
    """Tag Persian words with part-of-speech using Stanza."""
    from .fa.pos import tag_words

    df, col = read_wordlist(input_file, word_col=word_col)
    results = tag_words(df[col].astype(str).tolist())

    pos, lemma = [], []
    idx = 0
    for w in df[col].astype(str).tolist():
        if not str(w).strip():
            pos.append("")
            lemma.append("")
        else:
            r = results[idx]
            pos.append(r.pos_tag or "")
            lemma.append(r.lemma or "")
            idx += 1

    df["pos"] = pos
    df["lemma"] = lemma
    write_table(df, output_file)
    print(f"[green]✓[/green] Saved: {output_file}")


# English Commands

@en_app.command("g2p")
def en_g2p(
    input_file: str = typer.Argument(..., help="Input file"),
    output_file: str = typer.Argument(..., help="Output file"),
    word_col: Optional[str] = typer.Option(None, "--col", "-c", help="Word column name"),
):
    "Convert English text to phonemes (ARPAbet)."
    from .en.g2p import transcribe_words

    df, col = read_wordlist(input_file, word_col=word_col)
    words = df[col].astype(str).fillna("").tolist()
    results = transcribe_words(words)

    pron, err = [], []
    idx = 0
    for w in words:
        if not str(w).strip():
            pron.append("")
            err.append("")
        else:
            r = results[idx]
            pron.append(r.pronunciation)
            err.append(r.error or "")
            idx += 1

    df["pronunciation"] = pron
    df["g2p_error"] = err
    write_table(df, output_file)
    print(f"[green]✓[/green] Saved: {output_file}")


@en_app.command("syllables")
def en_syllables(
    input_file: str = typer.Argument(..., help="Input file"),
    output_file: str = typer.Argument(..., help="Output file"),
    word_col: Optional[str] = typer.Option(None, "--col", "-c", help="Word column name"),
):
    """Count syllables in English words."""
    from .en.syllables import count_syllables

    df, col = read_wordlist(input_file, word_col=word_col)
    df["syllables"] = df[col].astype(str).apply(count_syllables)
    write_table(df, output_file)
    print(f"[green]✓[/green] Saved: {output_file}")


@en_app.command("pos")
def en_pos(
    input_file: str = typer.Argument(..., help="Input file"),
    output_file: str = typer.Argument(..., help="Output file"),
    word_col: Optional[str] = typer.Option(None, "--col", "-c", help="Word column name"),
):
    """Tag English words with part-of-speech (spaCy)."""
    from .en.pos import tag_words

    df, col = read_wordlist(input_file, word_col=word_col)
    results = tag_words(df[col].astype(str).tolist())

    pos, tag, lemma = [], [], []
    idx = 0
    for w in df[col].astype(str).tolist():
        if not str(w).strip():
            pos.append("")
            tag.append("")
            lemma.append("")
        else:
            r = results[idx]
            pos.append(r.pos)
            tag.append(r.tag)
            lemma.append(r.lemma)
            idx += 1

    df["pos"] = pos
    df["tag"] = tag
    df["lemma"] = lemma
    write_table(df, output_file)
    print(f"[green]✓[/green] Saved: {output_file}")


# Japanese Commands

@ja_app.command("pos")
def ja_pos(
    input_file: str = typer.Argument(..., help="Input file"),
    output_file: str = typer.Argument(..., help="Output file"),
    word_col: Optional[str] = typer.Option(None, "--col", "-c", help="Word column name"),
    method: str = typer.Option("unidic", "--method", "-m", help="Method: unidic or stanza"),
):

    df, col = read_wordlist(input_file, word_col=word_col)
    words = df[col].astype(str).tolist()

    if method == "stanza":
        from .ja.pos_stanza import tag_pretokenized_with_stanza
        results = tag_pretokenized_with_stanza(words, download_if_missing=True)

        upos = []
        idx = 0
        for w in words:
            if not str(w).strip():
                upos.append("")
            else:
                upos.append(results[idx].upos)
                idx += 1
        df["pos"] = upos
    else:
        from .ja.pos_map import map_pos_to_english
        from .ja.pos_unidic import tag_with_unidic
        results = tag_with_unidic(words)

        pos1, lemma, pos_en = [], [], []
        idx = 0
        for w in words:
            if not str(w).strip():
                pos1.append("")
                lemma.append("")
                pos_en.append("")
            else:
                r = results[idx]
                pos1.append(r.pos1)
                lemma.append(r.lemma)
                pos_en.append(map_pos_to_english(r.pos1))
                idx += 1

        df["pos"] = pos1
        df["pos_english"] = pos_en
        df["lemma"] = lemma

    write_table(df, output_file)
    print(f"[green]✓[/green] Saved: {output_file}")


# Sampling Commands

@sample_app.command("stratified")
def sample_stratified(
    input_file: str = typer.Argument(..., help="Input file"),
    output_file: str = typer.Argument(..., help="Output file"),
    score_col: str = typer.Option(..., "--score", "-s", help="Score column (e.g., frequency)"),
    n: int = typer.Option(..., "--n", "-n", help="Sample size"),
    bins: int = typer.Option(3, "--bins", "-b", help="Number of bins"),
):
    "Stratified sampling by quantiles."
    from .sampling.stratified import stratified_sample_quantiles

    df = read_table(input_file)
    out, report = stratified_sample_quantiles(
        df, score_col=score_col, n_total=n, bins=bins
    )
    out = out.sort_values(by=score_col, ascending=False).reset_index(drop=True)
    write_table(out, output_file)
    print(f"[green]✓[/green] Sampled {report.total_sampled}/{n} rows → {output_file}")


@sample_app.command("shuffle")
def sample_shuffle(
    files: List[str] = typer.Argument(..., help="Input files (2+)"),
    output_dir: str = typer.Argument(..., help="Output directory"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
):
    """Shuffle rows across multiple files (keeps row correspondence)."""
    from .sampling.shuffle_rows import shuffle_corresponding_rows

    if len(files) < 2:
        raise typer.BadParameter("Need at least 2 files")

    dfs = [read_table(f) for f in files]
    out_dfs, report = shuffle_corresponding_rows(dfs, seed=seed)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for i, df in enumerate(out_dfs):
        name = Path(files[i]).stem
        ext = Path(files[i]).suffix or ".xlsx"
        write_table(df, out_path / f"{name}_shuffled{ext}")

    print(f"[green]✓[/green] Shuffled {report.n_files} files → {output_dir}/")


if __name__ == "__main__":
    app()
