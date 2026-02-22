"""Tests for lexprep.sampling.audit module (D4)."""

import json

import pandas as pd

from lexprep.sampling.audit import (
    audit_to_bytes,
    build_audit_dataframe,
    build_sampling_manifest_section,
)
from lexprep.sampling.stratified import (
    StratifiedSampleReport,
    StratifiedSampleResult,
    Stratum,
    stratified_sample_quantiles_full,
)


def _make_report():
    """Create a sample report for testing."""
    strata = [
        Stratum(
            index=0,
            lower=1.0,
            upper=5.0,
            count=30,
            mean=3.0,
            std=1.0,
            requested=10,
            sampled=10,
        ),
        Stratum(
            index=1,
            lower=5.0,
            upper=10.0,
            count=50,
            mean=7.5,
            std=1.5,
            requested=10,
            sampled=10,
        ),
        Stratum(
            index=2,
            lower=10.0,
            upper=20.0,
            count=20,
            mean=15.0,
            std=3.0,
            requested=10,
            sampled=10,
        ),
    ]
    return StratifiedSampleReport(
        total_population=100,
        total_requested=30,
        total_sampled=30,
        stratification_type="quantile",
        allocation_method="equal",
        strata_count=3,
        strata=strata,
        score_column="frequency",
        random_seed=42,
    )


class TestBuildAuditDataframe:
    def test_columns(self):
        report = _make_report()
        df = build_audit_dataframe(report)
        expected_cols = {
            "bin_id",
            "min",
            "max",
            "population_size",
            "selected_count",
            "excluded_count",
            "allocation_method",
        }
        assert set(df.columns) == expected_cols

    def test_row_count(self):
        report = _make_report()
        df = build_audit_dataframe(report)
        assert len(df) == 3

    def test_consistency_selected_plus_excluded(self):
        """selected_count + excluded_count should equal population_size."""
        report = _make_report()
        df = build_audit_dataframe(report)
        for _, row in df.iterrows():
            assert row["selected_count"] + row["excluded_count"] == row["population_size"]

    def test_total_population_matches(self):
        """Sum of population_size across bins should match total_population."""
        report = _make_report()
        df = build_audit_dataframe(report)
        assert df["population_size"].sum() == report.total_population

    def test_bin_range_values_are_numeric(self):
        report = _make_report()
        df = build_audit_dataframe(report)
        assert df["min"].dtype in ("float64", "int64")
        assert df["max"].dtype in ("float64", "int64")


class TestAuditToBytes:
    def test_returns_bytes(self):
        report = _make_report()
        data = audit_to_bytes(report)
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_is_valid_xlsx(self):
        report = _make_report()
        data = audit_to_bytes(report)
        import io

        df = pd.read_excel(io.BytesIO(data))
        assert "bin_id" in df.columns
        assert len(df) == 3


class TestBuildSamplingManifestSection:
    def test_structure(self):
        report = _make_report()
        section = build_sampling_manifest_section(report)
        assert section["variable"] == "frequency"
        assert section["binning"]["mode"] == "quantiles"
        assert len(section["binning"]["bins"]) == 3
        assert section["allocation"]["method"] == "equal"
        assert section["allocation"]["target_total_n"] == 30
        assert len(section["result"]["selected_per_bin"]) == 3

    def test_bin_range_values_not_rounded(self):
        """bin_range_min and bin_range_max must be exact floats."""
        report = _make_report()
        section = build_sampling_manifest_section(report)
        for b in section["binning"]["bins"]:
            assert isinstance(b["bin_range_min"], float)
            assert isinstance(b["bin_range_max"], float)

    def test_excluded_count_consistent(self):
        report = _make_report()
        section = build_sampling_manifest_section(report)
        assert section["result"]["excluded_count"] == report.total_population - report.total_sampled

    def test_json_serializable(self):
        report = _make_report()
        section = build_sampling_manifest_section(report)
        # Should not raise
        json.dumps(section)


class TestStratifiedSampleFull:
    def _make_df(self, n=100):
        import numpy as np

        np.random.seed(42)
        return pd.DataFrame(
            {
                "word": [f"w{i}" for i in range(n)],
                "freq": np.random.uniform(1, 100, n),
            }
        )

    def test_returns_result_type(self):
        df = self._make_df()
        result = stratified_sample_quantiles_full(
            df,
            score_col="freq",
            n_total=30,
            bins=3,
        )
        assert isinstance(result, StratifiedSampleResult)

    def test_bin_id_in_sample(self):
        df = self._make_df()
        result = stratified_sample_quantiles_full(
            df,
            score_col="freq",
            n_total=30,
            bins=3,
        )
        assert "bin_id" in result.sample_df.columns

    def test_bin_id_in_excluded(self):
        df = self._make_df()
        result = stratified_sample_quantiles_full(
            df,
            score_col="freq",
            n_total=30,
            bins=3,
        )
        assert "bin_id" in result.excluded_df.columns

    def test_sample_plus_excluded_equals_population(self):
        df = self._make_df()
        result = stratified_sample_quantiles_full(
            df,
            score_col="freq",
            n_total=30,
            bins=3,
        )
        total = len(result.sample_df) + len(result.excluded_df)
        assert total == len(df)

    def test_no_overlap_between_sample_and_excluded(self):
        df = self._make_df()
        result = stratified_sample_quantiles_full(
            df,
            score_col="freq",
            n_total=30,
            bins=3,
        )
        # Check no duplicate words
        sample_words = set(result.sample_df["word"].tolist())
        excluded_words = set(result.excluded_df["word"].tolist())
        assert len(sample_words & excluded_words) == 0

    def test_report_population_matches(self):
        df = self._make_df(50)
        result = stratified_sample_quantiles_full(
            df,
            score_col="freq",
            n_total=15,
            bins=3,
        )
        assert result.report.total_population == 50
