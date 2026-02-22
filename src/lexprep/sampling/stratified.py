"""Stratified Sampling Module
References:
- Cochran, W.G. (1977). Sampling Techniques (3rd ed.). Wiley.
- Lohr, S.L. (2019). Sampling: Design and Analysis (3rd ed.). CRC Press.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class AllocationMethod(Enum):
    """Sample allocation methods across strata."""
    EQUAL = "equal"           # Equal samples from each stratum
    PROPORTIONAL = "proportional"  # Proportional to stratum size
    OPTIMAL = "optimal"       # Neyman optimal allocation (minimizes variance)
    FIXED = "fixed"           # User-specified fixed counts


class StratificationType(Enum):
    """How strata are defined."""
    QUANTILE = "quantile"     # Automatic quantile-based bins
    CUSTOM = "custom"         # User-defined numeric ranges


@dataclass
class Stratum:
    """Represents a single stratum with its bounds and statistics."""
    index: int
    lower: float
    upper: float
    lower_inclusive: bool = True
    upper_inclusive: bool = True
    count: int = 0
    mean: float = 0.0
    std: float = 0.0
    requested: int = 0
    sampled: int = 0


@dataclass
class StratifiedSampleReport:
    """Detailed report of stratified sampling results."""
    total_population: int
    total_requested: int
    total_sampled: int
    stratification_type: str
    allocation_method: str
    strata_count: int
    strata: List[Stratum]
    score_column: str
    random_seed: int
    weights_used: Optional[Dict[int, float]] = None
    warnings: List[str] = None

    # Computed statistics
    coverage_rate: float = 0.0  # Proportion of population sampled
    design_effect: float = 1.0  # Efficiency vs simple random sampling

    def __post_init__(self):
        if self.total_population > 0:
            self.coverage_rate = self.total_sampled / self.total_population
        if self.warnings is None:
            self.warnings = []


@dataclass
class StratifiedSampleResult:
    """Full result of stratified sampling, including excluded rows."""

    sample_df: pd.DataFrame  # selected rows with bin_id column
    excluded_df: pd.DataFrame  # non-selected rows with bin_id column
    report: StratifiedSampleReport


@dataclass
class CustomRange:
    """User-defined range for custom stratification."""
    lower: float
    upper: float
    lower_inclusive: bool = True
    upper_inclusive: bool = True
    label: Optional[str] = None

    def contains(self, value: float) -> bool:
        """Check if a value falls within this range."""
        if pd.isna(value):
            return False
        lower_ok = value > self.lower if not self.lower_inclusive else value >= self.lower
        upper_ok = value < self.upper if not self.upper_inclusive else value <= self.upper
        return lower_ok and upper_ok

    def __str__(self) -> str:
        lb = "[" if self.lower_inclusive else "("
        ub = "]" if self.upper_inclusive else ")"
        return f"{lb}{self.lower}, {self.upper}{ub}"


def validate_custom_ranges(ranges: List[CustomRange]) -> Tuple[bool, str]:
    """
    Validate custom ranges for gaps and overlaps.

    Returns:
        (is_valid, error_message)
    """
    if not ranges:
        return False, "At least one range must be specified"

    # Sort by lower bound
    sorted_ranges = sorted(ranges, key=lambda r: r.lower)

    for i in range(len(sorted_ranges) - 1):
        current = sorted_ranges[i]
        next_range = sorted_ranges[i + 1]

        # Check for overlap
        if current.upper > next_range.lower:
            return False, f"Ranges overlap: {current} and {next_range}"

        # Check for overlap at boundary
        if current.upper == next_range.lower:
            if current.upper_inclusive and next_range.lower_inclusive:
                return False, f"Ranges overlap at boundary {current.upper}: both include this value"

    return True, ""


def _compute_allocation(
    strata_sizes: Dict[int, int],
    strata_stds: Dict[int, float],
    n_total: int,
    method: AllocationMethod,
    weights: Optional[Dict[int, float]] = None,
    fixed_counts: Optional[Dict[int, int]] = None
) -> Dict[int, int]:

    strata_indices = list(strata_sizes.keys())
    n_strata = len(strata_indices)
    n_population = sum(strata_sizes.values())

    if n_strata == 0 or n_population == 0:
        return {}

    allocation = {}

    if method == AllocationMethod.FIXED:
        if fixed_counts is None:
            raise ValueError("fixed_counts required for FIXED allocation method")
        for i in strata_indices:
            allocation[i] = min(fixed_counts.get(i, 0), strata_sizes[i])
        return allocation

    if method == AllocationMethod.EQUAL:
        # Equal allocation: n_h = n/n_strata
        base_n = n_total // n_strata
        remainder = n_total % n_strata

        for idx, i in enumerate(strata_indices):
            n_h = base_n + (1 if idx < remainder else 0)
            allocation[i] = min(n_h, strata_sizes[i])

    elif method == AllocationMethod.PROPORTIONAL:
        # Proportional allocation: n_h = n * (N_h / N)
        for i in strata_indices:
            n_h = int(round(n_total * strata_sizes[i] / n_population))
            allocation[i] = min(n_h, strata_sizes[i])

        # Adjust for rounding errors
        total_allocated = sum(allocation.values())
        diff = n_total - total_allocated

        if diff != 0:
            # Add/remove from largest strata
            sorted_strata = sorted(strata_indices, key=lambda x: strata_sizes[x], reverse=True)
            for i in sorted_strata:
                if diff == 0:
                    break
                if diff > 0 and allocation[i] < strata_sizes[i]:
                    allocation[i] += 1
                    diff -= 1
                elif diff < 0 and allocation[i] > 0:
                    allocation[i] -= 1
                    diff += 1

    elif method == AllocationMethod.OPTIMAL:
        # Neyman optimal allocation: n_h = n * (N_h * S_h) / sum(N_j * S_j)
        # Minimizes variance for fixed total sample size

        # Compute N_h * S_h for each stratum
        weighted_sizes = {}
        for i in strata_indices:
            s_h = strata_stds.get(i, 1.0)
            if s_h == 0:
                s_h = 0.001  # Avoid zero
            weighted_sizes[i] = strata_sizes[i] * s_h

        total_weighted = sum(weighted_sizes.values())

        if total_weighted == 0:
            # Fall back to proportional
            return _compute_allocation(strata_sizes, strata_stds, n_total,
                                       AllocationMethod.PROPORTIONAL)

        for i in strata_indices:
            n_h = int(round(n_total * weighted_sizes[i] / total_weighted))
            allocation[i] = min(n_h, strata_sizes[i])

        # Adjust for rounding
        total_allocated = sum(allocation.values())
        diff = n_total - total_allocated

        if diff != 0:
            sorted_strata = sorted(strata_indices, key=lambda x: weighted_sizes[x], reverse=True)
            for i in sorted_strata:
                if diff == 0:
                    break
                if diff > 0 and allocation[i] < strata_sizes[i]:
                    allocation[i] += 1
                    diff -= 1
                elif diff < 0 and allocation[i] > 0:
                    allocation[i] -= 1
                    diff += 1

    # Apply custom weights if provided (modifies allocation proportionally)
    if weights is not None and method != AllocationMethod.FIXED:
        total_weight = sum(weights.get(i, 1.0) for i in strata_indices)
        if total_weight > 0:
            weighted_allocation = {}
            for i in strata_indices:
                w = weights.get(i, 1.0) / total_weight
                n_h = int(round(n_total * w))
                weighted_allocation[i] = min(n_h, strata_sizes[i])

            # Adjust for rounding
            total_allocated = sum(weighted_allocation.values())
            diff = n_total - total_allocated
            if diff != 0:
                sorted_strata = sorted(
                    strata_indices, key=lambda x: weights.get(x, 1.0), reverse=True
                )
                for i in sorted_strata:
                    if diff == 0:
                        break
                    if diff > 0 and weighted_allocation[i] < strata_sizes[i]:
                        weighted_allocation[i] += 1
                        diff -= 1
                    elif diff < 0 and weighted_allocation[i] > 0:
                        weighted_allocation[i] -= 1
                        diff += 1

            allocation = weighted_allocation

    return allocation


def stratified_sample_quantiles(
    df: pd.DataFrame,
    *,
    score_col: str,
    n_total: int,
    bins: int = 3,
    allocation: Union[str, AllocationMethod] = AllocationMethod.EQUAL,
    weights: Optional[Dict[int, float]] = None,
    random_state: int = 19,
) -> Tuple[pd.DataFrame, StratifiedSampleReport]:

    if isinstance(allocation, str):
        allocation = AllocationMethod(allocation.lower())

    if score_col not in df.columns:
        raise ValueError(f"Column '{score_col}' not found. Available: {list(df.columns)}")

    # Convert to numeric
    scores = pd.to_numeric(df[score_col], errors="coerce")
    valid_mask = ~scores.isna()

    if valid_mask.sum() == 0:
        raise ValueError(f"Column '{score_col}' has no valid numeric values")

    # Create quantile bins
    try:
        bin_series, bin_edges = pd.qcut(
            scores[valid_mask],
            q=bins,
            labels=False,
            retbins=True,
            duplicates="drop"
        )
    except ValueError as e:
        raise ValueError(f"Cannot create {bins} quantile bins: {e}")

    # Create working dataframe
    df_work = df.copy()
    df_work["_stratum"] = pd.NA
    df_work.loc[valid_mask, "_stratum"] = bin_series.values
    df_work["_score"] = scores

    actual_bins = int(bin_series.max()) + 1 if len(bin_series) > 0 else 0

    # Compute stratum statistics
    strata = []
    strata_sizes = {}
    strata_stds = {}
    warnings = []

    for i in range(actual_bins):
        stratum_data = df_work[df_work["_stratum"] == i]
        stratum_scores = stratum_data["_score"]

        lower = bin_edges[i] if i < len(bin_edges) else float('-inf')
        upper = bin_edges[i + 1] if i + 1 < len(bin_edges) else float('inf')

        stratum = Stratum(
            index=i,
            lower=lower,
            upper=upper,
            lower_inclusive=True,
            upper_inclusive=(i == actual_bins - 1),
            count=len(stratum_data),
            mean=stratum_scores.mean() if len(stratum_data) > 0 else 0,
            std=stratum_scores.std() if len(stratum_data) > 1 else 0
        )
        strata.append(stratum)
        strata_sizes[i] = stratum.count
        strata_stds[i] = stratum.std

    # Check for dropped bins due to duplicates
    if actual_bins < bins:
        warnings.append(
            f"Requested {bins} bins, but only {actual_bins} could be created "
            f"due to duplicate values in '{score_col}'. Some bins were merged."
        )

    # Compute allocation
    allocated = _compute_allocation(
        strata_sizes, strata_stds, n_total, allocation, weights
    )

    # Sample from each stratum
    np.random.seed(random_state)
    sampled_parts = []

    for stratum in strata:
        i = stratum.index
        stratum_df = df_work[df_work["_stratum"] == i]
        n_h = allocated.get(i, 0)

        stratum.requested = n_h

        if n_h > 0 and len(stratum_df) > 0:
            sampled = stratum_df.sample(
                n=min(n_h, len(stratum_df)),
                random_state=random_state + i  # Different seed per stratum
            )
            sampled_parts.append(sampled)
            stratum.sampled = len(sampled)
        else:
            stratum.sampled = 0

    # Combine results
    if sampled_parts:
        result_df = pd.concat(sampled_parts, ignore_index=True)
        result_df = result_df.drop(columns=["_stratum", "_score"], errors="ignore")
    else:
        result_df = df.iloc[0:0].copy()

    # Check for under-allocation
    if len(result_df) < n_total:
        diff = n_total - len(result_df)
        warnings.append(
            f"Could not meet the requested sample size of {n_total}. "
            f"Only {len(result_df)} samples were available (shortfall: {diff})."
        )

    # Create report
    report = StratifiedSampleReport(
        total_population=len(df),
        total_requested=n_total,
        total_sampled=len(result_df),
        stratification_type=StratificationType.QUANTILE.value,
        allocation_method=allocation.value,
        strata_count=actual_bins,
        strata=strata,
        score_column=score_col,
        random_seed=random_state,
        weights_used=weights,
        warnings=warnings
    )

    return result_df, report


def stratified_sample_custom_ranges(
    df: pd.DataFrame,
    *,
    score_col: str,
    ranges: List[CustomRange],
    n_total: Optional[int] = None,
    allocation: Union[str, AllocationMethod] = AllocationMethod.EQUAL,
    weights: Optional[Dict[int, float]] = None,
    fixed_counts: Optional[Dict[int, int]] = None,
    random_state: int = 19,
) -> Tuple[pd.DataFrame, StratifiedSampleReport]:

    if isinstance(allocation, str):
        allocation = AllocationMethod(allocation.lower())

    # Validate inputs
    if score_col not in df.columns:
        raise ValueError(f"Column '{score_col}' not found. Available: {list(df.columns)}")

    is_valid, error_msg = validate_custom_ranges(ranges)
    if not is_valid:
        raise ValueError(f"Invalid ranges: {error_msg}")

    if allocation == AllocationMethod.FIXED:
        if fixed_counts is None:
            raise ValueError("fixed_counts required for FIXED allocation")
        n_total = sum(fixed_counts.values())
    elif n_total is None or n_total <= 0:
        raise ValueError("n_total must be positive for non-FIXED allocation")

    # Convert to numeric
    scores = pd.to_numeric(df[score_col], errors="coerce")

    # Assign rows to strata
    df_work = df.copy()
    df_work["_stratum"] = pd.NA
    df_work["_score"] = scores

    for idx, custom_range in enumerate(ranges):
        mask = df_work["_score"].apply(custom_range.contains)
        # Only assign if not already assigned (first matching range wins)
        unassigned = df_work["_stratum"].isna()
        df_work.loc[mask & unassigned, "_stratum"] = idx

    # Compute stratum statistics
    strata = []
    strata_sizes = {}
    strata_stds = {}
    warnings = []

    for idx, custom_range in enumerate(ranges):
        stratum_data = df_work[df_work["_stratum"] == idx]
        stratum_scores = stratum_data["_score"]

        stratum = Stratum(
            index=idx,
            lower=custom_range.lower,
            upper=custom_range.upper,
            lower_inclusive=custom_range.lower_inclusive,
            upper_inclusive=custom_range.upper_inclusive,
            count=len(stratum_data),
            mean=stratum_scores.mean() if len(stratum_data) > 0 else 0,
            std=stratum_scores.std() if len(stratum_data) > 1 else 0
        )
        strata.append(stratum)
        strata_sizes[idx] = stratum.count
        strata_stds[idx] = stratum.std

    # Compute allocation
    allocated = _compute_allocation(
        strata_sizes, strata_stds, n_total, allocation, weights, fixed_counts
    )

    # Sample from each stratum
    np.random.seed(random_state)
    sampled_parts = []

    for stratum in strata:
        i = stratum.index
        stratum_df = df_work[df_work["_stratum"] == i]
        n_h = allocated.get(i, 0)

        stratum.requested = n_h

        if n_h > 0 and len(stratum_df) > 0:
            actual_n = min(n_h, len(stratum_df))
            sampled = stratum_df.sample(
                n=actual_n,
                random_state=random_state + i
            )
            sampled_parts.append(sampled)
            stratum.sampled = len(sampled)
        else:
            stratum.sampled = 0

    # Combine results
    if sampled_parts:
        result_df = pd.concat(sampled_parts, ignore_index=True)
        result_df = result_df.drop(columns=["_stratum", "_score"], errors="ignore")
    else:
        result_df = df.iloc[0:0].copy()

    # Check for under-allocation
    if n_total is not None and len(result_df) < n_total:
        diff = n_total - len(result_df)
        warnings.append(
            f"Could not meet the requested sample size of {n_total}. "
            f"Only {len(result_df)} samples were available (shortfall: {diff})."
        )

    # Create report
    report = StratifiedSampleReport(
        total_population=len(df),
        total_requested=n_total,
        total_sampled=len(result_df),
        stratification_type=StratificationType.CUSTOM.value,
        allocation_method=allocation.value,
        strata_count=len(ranges),
        strata=strata,
        score_column=score_col,
        random_seed=random_state,
        weights_used=weights,
        warnings=warnings
    )

    return result_df, report


# Convenience function for backward compatibility
def stratified_sample(
    df: pd.DataFrame,
    *,
    score_col: str,
    n_total: int,
    bins: int = 3,
    allocation: str = "equal",
    weights: Optional[Dict[int, float]] = None,
    random_state: int = 19,
) -> Tuple[pd.DataFrame, StratifiedSampleReport]:
    """
    Convenience wrapper for stratified_sample_quantiles.
    Maintains backward compatibility with simpler API.
    """
    return stratified_sample_quantiles(
        df,
        score_col=score_col,
        n_total=n_total,
        bins=bins,
        allocation=allocation,
        weights=weights,
        random_state=random_state
    )


# ---------------------------------------------------------------------------
# Full variants — return sample + excluded + report (D4.1)
# ---------------------------------------------------------------------------

def _build_full_result(
    df_work: pd.DataFrame,
    sampled_parts: List[pd.DataFrame],
    original_df: pd.DataFrame,
    report: StratifiedSampleReport,
) -> StratifiedSampleResult:
    """Build a StratifiedSampleResult with bin_id on both sample and excluded."""
    # Collect sampled original indices
    sampled_indices = set()
    for part in sampled_parts:
        sampled_indices.update(part.index)

    # Build sample DataFrame with bin_id
    if sampled_parts:
        result_df = pd.concat(sampled_parts, ignore_index=False)
        result_df["bin_id"] = result_df["_stratum"].astype(int)
        result_df = result_df.drop(columns=["_stratum", "_score"], errors="ignore")
        result_df = result_df.reset_index(drop=True)
    else:
        result_df = original_df.iloc[0:0].copy()
        result_df["bin_id"] = pd.Series(dtype="int")

    # Build excluded DataFrame with bin_id
    excluded_mask = ~df_work.index.isin(sampled_indices)
    excluded_df = df_work[excluded_mask].copy()
    if "_stratum" in excluded_df.columns:
        excluded_df["bin_id"] = excluded_df["_stratum"].apply(
            lambda x: int(x) if pd.notna(x) else -1
        )
    else:
        excluded_df["bin_id"] = -1
    excluded_df = excluded_df.drop(columns=["_stratum", "_score"], errors="ignore")
    excluded_df = excluded_df.reset_index(drop=True)

    return StratifiedSampleResult(
        sample_df=result_df,
        excluded_df=excluded_df,
        report=report,
    )


def stratified_sample_quantiles_full(
    df: pd.DataFrame,
    *,
    score_col: str,
    n_total: int,
    bins: int = 3,
    allocation: Union[str, AllocationMethod] = AllocationMethod.EQUAL,
    weights: Optional[Dict[int, float]] = None,
    random_state: int = 19,
) -> StratifiedSampleResult:
    """Like ``stratified_sample_quantiles`` but returns sample + excluded + report."""
    if isinstance(allocation, str):
        allocation = AllocationMethod(allocation.lower())

    if score_col not in df.columns:
        raise ValueError(f"Column '{score_col}' not found. Available: {list(df.columns)}")

    scores = pd.to_numeric(df[score_col], errors="coerce")
    valid_mask = ~scores.isna()

    if valid_mask.sum() == 0:
        raise ValueError(f"Column '{score_col}' has no valid numeric values")

    try:
        bin_series, bin_edges = pd.qcut(
            scores[valid_mask], q=bins, labels=False, retbins=True, duplicates="drop"
        )
    except ValueError as e:
        raise ValueError(f"Cannot create {bins} quantile bins: {e}")

    df_work = df.copy()
    df_work["_stratum"] = pd.NA
    df_work.loc[valid_mask, "_stratum"] = bin_series.values
    df_work["_score"] = scores

    actual_bins = int(bin_series.max()) + 1 if len(bin_series) > 0 else 0

    strata = []
    strata_sizes = {}
    strata_stds = {}
    warnings = []

    for i in range(actual_bins):
        stratum_data = df_work[df_work["_stratum"] == i]
        stratum_scores = stratum_data["_score"]
        lower = bin_edges[i] if i < len(bin_edges) else float("-inf")
        upper = bin_edges[i + 1] if i + 1 < len(bin_edges) else float("inf")
        stratum = Stratum(
            index=i,
            lower=lower,
            upper=upper,
            lower_inclusive=True,
            upper_inclusive=(i == actual_bins - 1),
            count=len(stratum_data),
            mean=stratum_scores.mean() if len(stratum_data) > 0 else 0,
            std=stratum_scores.std() if len(stratum_data) > 1 else 0,
        )
        strata.append(stratum)
        strata_sizes[i] = stratum.count
        strata_stds[i] = stratum.std

    if actual_bins < bins:
        warnings.append(
            f"Requested {bins} bins, but only {actual_bins} could be created "
            f"due to duplicate values in '{score_col}'. Some bins were merged."
        )

    allocated = _compute_allocation(strata_sizes, strata_stds, n_total, allocation, weights)

    np.random.seed(random_state)
    sampled_parts = []

    for stratum in strata:
        i = stratum.index
        stratum_df = df_work[df_work["_stratum"] == i]
        n_h = allocated.get(i, 0)
        stratum.requested = n_h
        if n_h > 0 and len(stratum_df) > 0:
            sampled = stratum_df.sample(n=min(n_h, len(stratum_df)), random_state=random_state + i)
            sampled_parts.append(sampled)
            stratum.sampled = len(sampled)
        else:
            stratum.sampled = 0

    total_sampled = sum(len(p) for p in sampled_parts)
    if total_sampled < n_total:
        diff = n_total - total_sampled
        warnings.append(
            f"Could not meet the requested sample size of {n_total}. "
            f"Only {total_sampled} samples were available (shortfall: {diff})."
        )

    report = StratifiedSampleReport(
        total_population=len(df),
        total_requested=n_total,
        total_sampled=total_sampled,
        stratification_type=StratificationType.QUANTILE.value,
        allocation_method=allocation.value,
        strata_count=actual_bins,
        strata=strata,
        score_column=score_col,
        random_seed=random_state,
        weights_used=weights,
        warnings=warnings,
    )

    return _build_full_result(df_work, sampled_parts, df, report)


def stratified_sample_custom_ranges_full(
    df: pd.DataFrame,
    *,
    score_col: str,
    ranges: List[CustomRange],
    n_total: Optional[int] = None,
    allocation: Union[str, AllocationMethod] = AllocationMethod.EQUAL,
    weights: Optional[Dict[int, float]] = None,
    fixed_counts: Optional[Dict[int, int]] = None,
    random_state: int = 19,
) -> StratifiedSampleResult:
    """Like ``stratified_sample_custom_ranges`` but returns sample + excluded + report."""
    if isinstance(allocation, str):
        allocation = AllocationMethod(allocation.lower())

    if score_col not in df.columns:
        raise ValueError(f"Column '{score_col}' not found. Available: {list(df.columns)}")

    is_valid, error_msg = validate_custom_ranges(ranges)
    if not is_valid:
        raise ValueError(f"Invalid ranges: {error_msg}")

    if allocation == AllocationMethod.FIXED:
        if fixed_counts is None:
            raise ValueError("fixed_counts required for FIXED allocation")
        n_total = sum(fixed_counts.values())
    elif n_total is None or n_total <= 0:
        raise ValueError("n_total must be positive for non-FIXED allocation")

    scores = pd.to_numeric(df[score_col], errors="coerce")

    df_work = df.copy()
    df_work["_stratum"] = pd.NA
    df_work["_score"] = scores

    for idx, custom_range in enumerate(ranges):
        mask = df_work["_score"].apply(custom_range.contains)
        unassigned = df_work["_stratum"].isna()
        df_work.loc[mask & unassigned, "_stratum"] = idx

    strata = []
    strata_sizes = {}
    strata_stds = {}
    warnings = []

    for idx, custom_range in enumerate(ranges):
        stratum_data = df_work[df_work["_stratum"] == idx]
        stratum_scores = stratum_data["_score"]
        stratum = Stratum(
            index=idx,
            lower=custom_range.lower,
            upper=custom_range.upper,
            lower_inclusive=custom_range.lower_inclusive,
            upper_inclusive=custom_range.upper_inclusive,
            count=len(stratum_data),
            mean=stratum_scores.mean() if len(stratum_data) > 0 else 0,
            std=stratum_scores.std() if len(stratum_data) > 1 else 0,
        )
        strata.append(stratum)
        strata_sizes[idx] = stratum.count
        strata_stds[idx] = stratum.std

    allocated = _compute_allocation(
        strata_sizes, strata_stds, n_total, allocation, weights, fixed_counts
    )

    np.random.seed(random_state)
    sampled_parts = []

    for stratum in strata:
        i = stratum.index
        stratum_df = df_work[df_work["_stratum"] == i]
        n_h = allocated.get(i, 0)
        stratum.requested = n_h
        if n_h > 0 and len(stratum_df) > 0:
            actual_n = min(n_h, len(stratum_df))
            sampled = stratum_df.sample(n=actual_n, random_state=random_state + i)
            sampled_parts.append(sampled)
            stratum.sampled = len(sampled)
        else:
            stratum.sampled = 0

    total_sampled = sum(len(p) for p in sampled_parts)
    if n_total is not None and total_sampled < n_total:
        diff = n_total - total_sampled
        warnings.append(
            f"Could not meet the requested sample size of {n_total}. "
            f"Only {total_sampled} samples were available (shortfall: {diff})."
        )

    report = StratifiedSampleReport(
        total_population=len(df),
        total_requested=n_total,
        total_sampled=total_sampled,
        stratification_type=StratificationType.CUSTOM.value,
        allocation_method=allocation.value,
        strata_count=len(ranges),
        strata=strata,
        score_column=score_col,
        random_seed=random_state,
        weights_used=weights,
        warnings=warnings,
    )

    return _build_full_result(df_work, sampled_parts, df, report)
