"""Sampling audit file and manifest section generation (D4.2, D4.3)."""

from __future__ import annotations

import io
from typing import Any, Dict, List

import pandas as pd

from .stratified import StratifiedSampleReport


def build_audit_dataframe(report: StratifiedSampleReport) -> pd.DataFrame:
    """Build one-row-per-bin audit summary table.

    Columns: bin_id, min, max, population_size, selected_count,
    excluded_count, allocation_method.
    """
    rows: List[Dict[str, Any]] = []
    for s in report.strata:
        rows.append(
            {
                "bin_id": s.index,
                "min": s.lower,
                "max": s.upper,
                "population_size": s.count,
                "selected_count": s.sampled,
                "excluded_count": s.count - s.sampled,
                "allocation_method": report.allocation_method,
            }
        )
    return pd.DataFrame(rows)


def audit_to_bytes(report: StratifiedSampleReport) -> bytes:
    """Generate ``sampling_audit.xlsx`` as bytes."""
    df = build_audit_dataframe(report)
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    return buf.getvalue()


def build_sampling_manifest_section(
    report: StratifiedSampleReport,
) -> Dict[str, Any]:
    """Build the ``sampling`` section for ``run_manifest.json``.

    Values such as ``bin_range_min`` / ``bin_range_max`` are kept as raw
    floats (not rounded) for methodological accuracy.
    """
    bins: List[Dict[str, Any]] = []
    selected_per_bin: List[int] = []
    for s in report.strata:
        bins.append(
            {
                "bin_id": s.index,
                "bin_range_min": float(s.lower),
                "bin_range_max": float(s.upper),
                "population_size": s.count,
            }
        )
        selected_per_bin.append(s.sampled)

    return {
        "variable": report.score_column,
        "binning": {
            "mode": ("quantiles" if report.stratification_type == "quantile" else "custom_ranges"),
            "bins": bins,
        },
        "allocation": {
            "method": report.allocation_method,
            "formula_reference": "Cochran 1977",
            "target_total_n": report.total_requested,
        },
        "result": {
            "selected_per_bin": selected_per_bin,
            "excluded_count": report.total_population - report.total_sampled,
        },
    }
