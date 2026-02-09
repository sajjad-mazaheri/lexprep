""" tests for lexprep.sampling.stratified module."""

import pandas as pd
import pytest

from lexprep.sampling.stratified import (
    AllocationMethod,
    CustomRange,
    StratifiedSampleReport,
    _compute_allocation,
    stratified_sample,
    stratified_sample_custom_ranges,
    stratified_sample_quantiles,
    validate_custom_ranges,
)


# fixtures

@pytest.fixture
def df_100():
    """DataFrame with 100 rows and a uniform score column."""
    return pd.DataFrame({"word": [f"w{i}" for i in range(100)], "score": list(range(100))})


@pytest.fixture
def df_small():
    """Small DataFrame for edge-case testing."""
    return pd.DataFrame({"word": ["a", "b", "c", "d", "e"], "score": [10, 20, 30, 40, 50]})


# validate_custom_ranges


class TestValidateCustomRanges:
    def test_empty_ranges(self):
        ok, msg = validate_custom_ranges([])
        assert not ok
        assert "at least one" in msg.lower()

    def test_single_valid(self):
        ok, _ = validate_custom_ranges([CustomRange(0, 10)])
        assert ok

    def test_no_overlap(self):
        ok, _ = validate_custom_ranges([
            CustomRange(0, 10, upper_inclusive=False),
            CustomRange(10, 20, lower_inclusive=True),
        ])
        assert ok

    def test_overlap_detected(self):
        ok, msg = validate_custom_ranges([
            CustomRange(0, 15),
            CustomRange(10, 20),
        ])
        assert not ok
        assert "overlap" in msg.lower()

    def test_boundary_overlap(self):
        ok, msg = validate_custom_ranges([
            CustomRange(0, 10, upper_inclusive=True),
            CustomRange(10, 20, lower_inclusive=True),
        ])
        assert not ok
        assert "boundary" in msg.lower()


# compute_allocation


class TestComputeAllocation:
    def test_equal(self):
        sizes = {0: 50, 1: 50}
        stds = {0: 1.0, 1: 1.0}
        alloc = _compute_allocation(sizes, stds, 20, AllocationMethod.EQUAL)
        assert sum(alloc.values()) == 20
        assert alloc[0] == 10
        assert alloc[1] == 10

    def test_proportional(self):
        sizes = {0: 75, 1: 25}
        stds = {0: 1.0, 1: 1.0}
        alloc = _compute_allocation(sizes, stds, 20, AllocationMethod.PROPORTIONAL)
        assert sum(alloc.values()) == 20
        assert alloc[0] > alloc[1]

    def test_fixed(self):
        sizes = {0: 50, 1: 50}
        stds = {0: 1.0, 1: 1.0}
        fixed = {0: 5, 1: 15}
        alloc = _compute_allocation(sizes, stds, 20, AllocationMethod.FIXED, fixed_counts=fixed)
        assert alloc[0] == 5
        assert alloc[1] == 15

    def test_fixed_capped_by_stratum_size(self):
        sizes = {0: 3, 1: 50}
        stds = {0: 1.0, 1: 1.0}
        fixed = {0: 10, 1: 10}
        alloc = _compute_allocation(sizes, stds, 20, AllocationMethod.FIXED, fixed_counts=fixed)
        assert alloc[0] == 3  # capped to actual size

    def test_optimal(self):
        sizes = {0: 50, 1: 50}
        stds = {0: 10.0, 1: 1.0}
        alloc = _compute_allocation(sizes, stds, 20, AllocationMethod.OPTIMAL)
        assert sum(alloc.values()) == 20
        assert alloc[0] > alloc[1]  # higher std → more samples

    def test_empty_strata(self):
        alloc = _compute_allocation({}, {}, 10, AllocationMethod.EQUAL)
        assert alloc == {}

    def test_fixed_without_counts_raises(self):
        with pytest.raises(ValueError, match="fixed_counts"):
            _compute_allocation({0: 10}, {0: 1.0}, 10, AllocationMethod.FIXED)


# stratified_sample_quantiles


class TestStratifiedSampleQuantiles:
    def test_basic_sample(self, df_100):
        out, report = stratified_sample_quantiles(df_100, score_col="score", n_total=30, bins=3)
        assert len(out) == 30
        assert report.total_sampled == 30
        assert report.strata_count == 3

    def test_reproducible(self, df_100):
        out1, _ = stratified_sample_quantiles(df_100, score_col="score", n_total=20, bins=2, random_state=42)
        out2, _ = stratified_sample_quantiles(df_100, score_col="score", n_total=20, bins=2, random_state=42)
        assert out1["word"].tolist() == out2["word"].tolist()

    def test_different_seeds_differ(self, df_100):
        out1, _ = stratified_sample_quantiles(df_100, score_col="score", n_total=20, bins=2, random_state=1)
        out2, _ = stratified_sample_quantiles(df_100, score_col="score", n_total=20, bins=2, random_state=99)
        assert out1["word"].tolist() != out2["word"].tolist()

    def test_missing_column_raises(self, df_100):
        with pytest.raises(ValueError, match="not found"):
            stratified_sample_quantiles(df_100, score_col="nonexistent", n_total=10, bins=2)

    def test_non_numeric_column_raises(self):
        df = pd.DataFrame({"score": ["a", "b", "c"]})
        with pytest.raises(ValueError, match="no valid numeric"):
            stratified_sample_quantiles(df, score_col="score", n_total=2, bins=2)

    def test_report_fields(self, df_100):
        _, report = stratified_sample_quantiles(df_100, score_col="score", n_total=15, bins=3)
        assert isinstance(report, StratifiedSampleReport)
        assert report.total_population == 100
        assert report.total_requested == 15
        assert report.score_column == "score"
        assert report.stratification_type == "quantile"
        assert report.coverage_rate == report.total_sampled / 100

    def test_under_allocation_warning(self, df_small):
        _, report = stratified_sample_quantiles(df_small, score_col="score", n_total=100, bins=2)
        assert report.total_sampled < 100
        assert any("shortfall" in w.lower() for w in report.warnings)

    def test_proportional_allocation(self, df_100):
        out, report = stratified_sample_quantiles(
            df_100, score_col="score", n_total=30, bins=3, allocation="proportional"
        )
        assert report.allocation_method == "proportional"
        assert report.total_sampled == 30

    def test_output_has_no_internal_columns(self, df_100):
        out, _ = stratified_sample_quantiles(df_100, score_col="score", n_total=10, bins=2)
        assert "_stratum" not in out.columns
        assert "_score" not in out.columns


# stratified_sample_custom_ranges


class TestStratifiedSampleCustomRanges:
    def test_basic_custom(self, df_100):
        ranges = [
            CustomRange(0, 33),
            CustomRange(34, 66),
            CustomRange(67, 99),
        ]
        out, report = stratified_sample_custom_ranges(
            df_100, score_col="score", ranges=ranges, n_total=15
        )
        assert report.total_sampled <= 15
        assert report.stratification_type == "custom"
        assert report.strata_count == 3

    def test_fixed_allocation(self, df_100):
        ranges = [CustomRange(0, 50), CustomRange(51, 99)]
        fixed = {0: 5, 1: 10}
        out, report = stratified_sample_custom_ranges(
            df_100, score_col="score", ranges=ranges,
            allocation="fixed", fixed_counts=fixed
        )
        assert report.allocation_method == "fixed"
        assert report.total_sampled <= 15

    def test_invalid_ranges_raises(self, df_100):
        ranges = [CustomRange(0, 60), CustomRange(30, 90)]  # overlap
        with pytest.raises(ValueError, match="Invalid ranges"):
            stratified_sample_custom_ranges(
                df_100, score_col="score", ranges=ranges, n_total=10
            )


# backward compat wrapper


class TestStratifiedSampleCompat:
    def test_convenience_wrapper(self, df_100):
        out, report = stratified_sample(df_100, score_col="score", n_total=20, bins=2)
        assert report.total_sampled == 20
