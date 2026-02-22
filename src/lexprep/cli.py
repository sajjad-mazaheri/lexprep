from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer
from rich import print

from .io.files import read_table, read_wordlist
from .manifest import build_manifest, get_libraries_for_tool, utc_now
from .packaging import build_zip

app = typer.Typer(
    add_completion=False,
    help="lexprep - wordlist preparation toolkit",
    no_args_is_help=True,
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _input_ext(path: str) -> str:
    """Get extension without dot (csv, tsv, xlsx)."""
    ext = Path(path).suffix.lstrip(".").lower()
    return ext if ext in ("csv", "tsv") else "xlsx"


def _zip_output_path(output_file: str) -> Path:
    """Replace user's extension with .zip."""
    return Path(output_file).with_suffix(".zip")


def _write_zip(zip_bytes: bytes, zip_name: str, output_file: str) -> None:
    """Write ZIP bytes and print success."""
    out = _zip_output_path(output_file)
    out.write_bytes(zip_bytes)
    print(f"[green]✓[/green] Saved: {out}")


# ---------------------------------------------------------------------------
# Persian Commands
# ---------------------------------------------------------------------------


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

    added = ["pronunciation", "g2p_error"]
    ts = utc_now()
    manifest = build_manifest(
        tool_key="g2p",
        language="fa",
        original_filename=Path(input_file).name,
        file_type=Path(input_file).suffix.lstrip("."),
        row_count=len(df),
        column_mapping={"word_column": col},
        added_columns=added,
        libraries=get_libraries_for_tool("fa", "g2p"),
        timestamp=ts,
    )
    zip_bytes, zip_name = build_zip(
        manifest=manifest,
        main_df=df,
        input_basename=Path(input_file).stem,
        output_ext=_input_ext(input_file),
    )
    _write_zip(zip_bytes, zip_name, output_file)


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

    tool_key = "syllables"
    added = []

    if with_g2p:
        from .fa.g2p import transcribe_words

        tool_key = "syllables_phonetic"
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
        added = ["pronunciation", "syllables"]
    elif method == "phonetic":
        tool_key = "syllables_phonetic"
        df = read_table(input_file)
        col = word_col or "word"
        if pron_col not in df.columns:
            raise typer.BadParameter(f"Column '{pron_col}' not found")
        df["syllables"] = df[pron_col].astype(str).apply(count_syllables_from_pronunciation)
        added = ["syllables"]
    else:
        df, col = read_wordlist(input_file, word_col=word_col)
        sylls, counts = [], []
        for w in df[col].astype(str).tolist():
            s, c = syllabify_orthographic(w)
            sylls.append(s)
            counts.append(c)
        df["syllabified"] = sylls
        df["syllables"] = counts
        added = ["syllabified", "syllables"]

    # Summary stats
    syll_col = df["syllables"] if "syllables" in df.columns else None
    summary = None
    if syll_col is not None:
        summary = {
            "syllable_distribution": syll_col.value_counts().sort_index().to_dict()
        }

    ts = utc_now()
    manifest = build_manifest(
        tool_key=tool_key,
        language="fa",
        original_filename=Path(input_file).name,
        file_type=Path(input_file).suffix.lstrip("."),
        row_count=len(df),
        column_mapping={"word_column": col if 'col' in dir() else "word"},
        added_columns=added,
        libraries=get_libraries_for_tool("fa", tool_key),
        timestamp=ts,
        summary=summary,
    )
    zip_bytes, zip_name = build_zip(
        manifest=manifest,
        main_df=df,
        input_basename=Path(input_file).stem,
        output_ext=_input_ext(input_file),
    )
    _write_zip(zip_bytes, zip_name, output_file)


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

    pos_list, lemma = [], []
    idx = 0
    for w in df[col].astype(str).tolist():
        if not str(w).strip():
            pos_list.append("")
            lemma.append("")
        else:
            r = results[idx]
            pos_list.append(r.pos_tag or "")
            lemma.append(r.lemma or "")
            idx += 1

    df["pos"] = pos_list
    df["lemma"] = lemma

    added = ["pos", "lemma"]
    summary = {"pos_distribution": df["pos"].value_counts().to_dict()}

    ts = utc_now()
    manifest = build_manifest(
        tool_key="pos",
        language="fa",
        original_filename=Path(input_file).name,
        file_type=Path(input_file).suffix.lstrip("."),
        row_count=len(df),
        column_mapping={"word_column": col},
        added_columns=added,
        libraries=get_libraries_for_tool("fa", "pos"),
        timestamp=ts,
        summary=summary,
    )
    zip_bytes, zip_name = build_zip(
        manifest=manifest,
        main_df=df,
        input_basename=Path(input_file).stem,
        output_ext=_input_ext(input_file),
    )
    _write_zip(zip_bytes, zip_name, output_file)


# ---------------------------------------------------------------------------
# English Commands
# ---------------------------------------------------------------------------


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

    added = ["pronunciation", "g2p_error"]
    ts = utc_now()
    manifest = build_manifest(
        tool_key="g2p",
        language="en",
        original_filename=Path(input_file).name,
        file_type=Path(input_file).suffix.lstrip("."),
        row_count=len(df),
        column_mapping={"word_column": col},
        added_columns=added,
        libraries=get_libraries_for_tool("en", "g2p"),
        timestamp=ts,
    )
    zip_bytes, zip_name = build_zip(
        manifest=manifest,
        main_df=df,
        input_basename=Path(input_file).stem,
        output_ext=_input_ext(input_file),
    )
    _write_zip(zip_bytes, zip_name, output_file)


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

    added = ["syllables"]
    summary = {
        "syllable_distribution": df["syllables"].value_counts().sort_index().to_dict()
    }

    ts = utc_now()
    manifest = build_manifest(
        tool_key="syllables",
        language="en",
        original_filename=Path(input_file).name,
        file_type=Path(input_file).suffix.lstrip("."),
        row_count=len(df),
        column_mapping={"word_column": col},
        added_columns=added,
        libraries=get_libraries_for_tool("en", "syllables"),
        timestamp=ts,
        summary=summary,
    )
    zip_bytes, zip_name = build_zip(
        manifest=manifest,
        main_df=df,
        input_basename=Path(input_file).stem,
        output_ext=_input_ext(input_file),
    )
    _write_zip(zip_bytes, zip_name, output_file)


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

    pos_list, tag_list, lemma = [], [], []
    idx = 0
    for w in df[col].astype(str).tolist():
        if not str(w).strip():
            pos_list.append("")
            tag_list.append("")
            lemma.append("")
        else:
            r = results[idx]
            pos_list.append(r.pos)
            tag_list.append(r.tag)
            lemma.append(r.lemma)
            idx += 1

    df["pos"] = pos_list
    df["tag"] = tag_list
    df["lemma"] = lemma

    added = ["pos", "tag", "lemma"]
    summary = {"pos_distribution": df["pos"].value_counts().to_dict()}

    ts = utc_now()
    manifest = build_manifest(
        tool_key="pos",
        language="en",
        original_filename=Path(input_file).name,
        file_type=Path(input_file).suffix.lstrip("."),
        row_count=len(df),
        column_mapping={"word_column": col},
        added_columns=added,
        libraries=get_libraries_for_tool("en", "pos"),
        timestamp=ts,
        summary=summary,
    )
    zip_bytes, zip_name = build_zip(
        manifest=manifest,
        main_df=df,
        input_basename=Path(input_file).stem,
        output_ext=_input_ext(input_file),
    )
    _write_zip(zip_bytes, zip_name, output_file)


# ---------------------------------------------------------------------------
# Japanese Commands
# ---------------------------------------------------------------------------


@ja_app.command("pos")
def ja_pos(
    input_file: str = typer.Argument(..., help="Input file"),
    output_file: str = typer.Argument(..., help="Output file"),
    word_col: Optional[str] = typer.Option(None, "--col", "-c", help="Word column name"),
    method: str = typer.Option("unidic", "--method", "-m", help="Method: unidic or stanza"),
):
    df, col = read_wordlist(input_file, word_col=word_col)
    words = df[col].astype(str).tolist()
    tool_key = "pos_stanza" if method == "stanza" else "pos_unidic"

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
        added = ["pos"]
    else:
        from .ja.pos_map import map_pos_to_english
        from .ja.pos_unidic import tag_with_unidic

        unidic_results = tag_with_unidic(words)
        pos1, lemma, pos_en = [], [], []
        idx = 0
        for w in words:
            if not str(w).strip():
                pos1.append("")
                lemma.append("")
                pos_en.append("")
            else:
                r = unidic_results[idx]
                pos1.append(r.pos1)
                lemma.append(r.lemma)
                pos_en.append(map_pos_to_english(r.pos1))
                idx += 1

        df["pos"] = pos1
        df["pos_english"] = pos_en
        df["lemma"] = lemma
        added = ["pos", "pos_english", "lemma"]

    summary = {"pos_distribution": df["pos"].value_counts().to_dict()}

    ts = utc_now()
    manifest = build_manifest(
        tool_key=tool_key,
        language="ja",
        original_filename=Path(input_file).name,
        file_type=Path(input_file).suffix.lstrip("."),
        row_count=len(df),
        column_mapping={"word_column": col},
        added_columns=added,
        libraries=get_libraries_for_tool("ja", tool_key),
        timestamp=ts,
        summary=summary,
    )
    zip_bytes, zip_name = build_zip(
        manifest=manifest,
        main_df=df,
        input_basename=Path(input_file).stem,
        output_ext=_input_ext(input_file),
    )
    _write_zip(zip_bytes, zip_name, output_file)


# ---------------------------------------------------------------------------
# Length Command (language-agnostic — D3)
# ---------------------------------------------------------------------------


@app.command("length")
def length_cmd(
    input_file: str = typer.Argument(..., help="Input file"),
    output_file: str = typer.Argument(..., help="Output file"),
    word_col: Optional[str] = typer.Option(None, "--col", "-c", help="Word column name"),
    language: str = typer.Option("en", "--language", "-l", help="Language code (fa/en/ja)"),
):
    """Add length_chars column (Unicode codepoint count)."""
    from .length import LENGTH_METHOD, compute_length_chars, length_distribution

    df, col = read_wordlist(input_file, word_col=word_col)
    words = df[col].astype(str).fillna("").tolist()
    lengths = compute_length_chars(words)
    df["length_chars"] = lengths

    dist = length_distribution(lengths)
    summary: dict[str, object] = {}
    if dist:
        summary["length_distribution"] = dist.to_dict()
    summary["length_method"] = LENGTH_METHOD

    added = ["length_chars"]
    ts = utc_now()
    manifest = build_manifest(
        tool_key="length",
        language=language,
        original_filename=Path(input_file).name,
        file_type=Path(input_file).suffix.lstrip("."),
        row_count=len(df),
        column_mapping={"word_column": col},
        added_columns=added,
        libraries=get_libraries_for_tool(language, "length"),
        timestamp=ts,
        summary=summary,
    )
    zip_bytes, zip_name = build_zip(
        manifest=manifest,
        main_df=df,
        input_basename=Path(input_file).stem,
        output_ext=_input_ext(input_file),
    )
    _write_zip(zip_bytes, zip_name, output_file)


# ---------------------------------------------------------------------------
# Sampling Commands
# ---------------------------------------------------------------------------


@sample_app.command("stratified")
def sample_stratified(
    input_file: str = typer.Argument(..., help="Input file"),
    output_file: str = typer.Argument(..., help="Output file"),
    score_col: str = typer.Option(..., "--score", "-s", help="Score column (e.g., frequency)"),
    n: int = typer.Option(..., "--n", "-n", help="Sample size"),
    bins: int = typer.Option(3, "--bins", "-b", help="Number of bins"),
    seed: int = typer.Option(19, "--seed", help="Random seed"),
):
    "Stratified sampling by quantiles."
    from .sampling.audit import audit_to_bytes, build_sampling_manifest_section
    from .sampling.stratified import stratified_sample_quantiles_full

    df = read_table(input_file)
    result = stratified_sample_quantiles_full(
        df, score_col=score_col, n_total=n, bins=bins, random_state=seed
    )
    sample_df = result.sample_df.sort_values(by=score_col, ascending=False).reset_index(drop=True)

    audit_bytes_data = audit_to_bytes(result.report)
    sampling_section = build_sampling_manifest_section(result.report)

    ts = utc_now()
    manifest = build_manifest(
        tool_key="stratified",
        language=None,
        original_filename=Path(input_file).name,
        file_type=Path(input_file).suffix.lstrip("."),
        row_count=len(df),
        column_mapping={"score_column": score_col},
        added_columns=["bin_id"],
        libraries=[],
        timestamp=ts,
        reproducibility={
            "seed": seed,
            "parameters": {"mode": "quantiles", "bins": bins, "n_total": n},
        },
        sampling=sampling_section,
    )
    zip_bytes, zip_name = build_zip(
        manifest=manifest,
        main_df=sample_df,
        input_basename=Path(input_file).stem,
        output_ext="xlsx",
        is_sampling=True,
        excluded_df=result.excluded_df,
        audit_bytes=audit_bytes_data,
    )
    _write_zip(zip_bytes, zip_name, output_file)
    print(
        f"  Sampled {result.report.total_sampled}/{n} rows"
    )


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

    # Build extra files for ZIP
    extra_files = []
    for i, out_df in enumerate(out_dfs):
        name = Path(files[i]).stem
        ext = Path(files[i]).suffix.lstrip(".").lower()
        if ext not in ("csv", "tsv"):
            ext = "xlsx"
        fname = f"{name}_shuffled.{ext}"
        from .packaging import _df_to_bytes

        extra_files.append((fname, _df_to_bytes(out_df, ext)))

    ts = utc_now()
    import pandas as pd

    manifest = build_manifest(
        tool_key="shuffle",
        language=None,
        original_filename=", ".join(Path(f).name for f in files),
        file_type="multiple",
        row_count=report.n_rows,
        column_mapping={"used_columns": ", ".join(report.used_columns)},
        added_columns=[],
        libraries=[],
        timestamp=ts,
        reproducibility={
            "seed": seed,
            "parameters": {
                "number_of_files": report.n_files,
                "row_count": report.n_rows,
                "shuffle_mode": "synchronized_row_permutation",
            },
        },
    )

    # For shuffle we use a dummy empty main_df since all files are in extra_files
    zip_bytes, zip_name = build_zip(
        manifest=manifest,
        main_df=pd.DataFrame(),
        input_basename="shuffle",
        output_ext="xlsx",
        extra_files=extra_files,
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    zip_path = out_path / zip_name
    zip_path.write_bytes(zip_bytes)
    print(f"[green]✓[/green] Shuffled {report.n_files} files → {zip_path}")


if __name__ == "__main__":
    app()
