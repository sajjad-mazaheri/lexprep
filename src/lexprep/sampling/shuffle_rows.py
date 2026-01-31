from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


@dataclass
class ShuffleReport:
    n_files: int
    n_rows: int
    used_columns: List[str]


def shuffle_corresponding_rows(
    dfs: List[pd.DataFrame],
    *,
    seed: int = 19,
    n_columns: Optional[int] = None,
) -> tuple[List[pd.DataFrame], ShuffleReport]:

    if len(dfs) < 2:
        raise ValueError("Provide at least 2 files")

    n_rows = dfs[0].shape[0]
    for j, df in enumerate(dfs):
        if df.shape[0] != n_rows:
            raise ValueError(f"Row count mismatch: file 0 has {n_rows}, file {j} has {df.shape[0]}")

    # Align columns
    base_cols = list(dfs[0].columns)
    if n_columns is not None:
        base_cols = base_cols[: int(n_columns)]

    aligned = [df.loc[:, base_cols].copy() for df in dfs]

    import random

    rng = random.Random(seed)
    out_dfs = [pd.DataFrame(columns=base_cols) for _ in aligned]

    for i in range(n_rows):
        row_group = [aligned[j].iloc[i].copy() for j in range(len(aligned))]
        rng.shuffle(row_group)
        for j in range(len(aligned)):
            out_dfs[j].loc[i] = row_group[j].values

    report = ShuffleReport(n_files=len(dfs), n_rows=n_rows, used_columns=base_cols)
    return out_dfs, report
