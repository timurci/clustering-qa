"""Unit tests for the clustering_agreement pipeline nodes."""

import math

import numpy as np
import polars as pl
import pytest

from clustering_qa.datasets.feature_significance import FeatureSignificance
from clustering_qa.datasets.feature_table import FeatureTable, SampleLabels
from clustering_qa.pipelines.clustering_agreement.nodes import (
    intersect_significant_features,
    kruskal_wallis_per_feature,
)


def _make_feature_table(rng: np.random.Generator, n: int) -> FeatureTable:
    """Builds a FeatureTable with three features and clear group effects."""
    sample_ids = np.arange(n, dtype=np.uint32)
    separable = np.where(
        np.arange(n) % 2 == 0,
        rng.normal(loc=0.0, scale=1.0, size=n),
        rng.normal(loc=5.0, scale=1.0, size=n),
    )
    noisy = rng.normal(loc=0.0, scale=1.0, size=n)
    constant = np.full(n, 3.14, dtype=np.float64)
    df = pl.DataFrame(
        {
            "sample_id": sample_ids,
            "separable": separable,
            "noisy": noisy,
            "constant": constant,
        }
    )
    return FeatureTable(data=df)


def _make_labels(n: int) -> SampleLabels:
    """Alternating binary labels for n samples."""
    sample_ids = np.arange(n, dtype=np.uint32)
    labels = np.array([str(i % 2) for i in range(n)])
    df = pl.DataFrame({"sample_id": sample_ids, "label": labels})
    return SampleLabels(data=df)


class TestKruskalWallisPerFeature:
    """Tests for the ``kruskal_wallis_per_feature`` node function."""

    def test_returns_one_row_per_feature(self) -> None:
        rng = np.random.default_rng(0)
        features = _make_feature_table(rng, n=200)
        labels = _make_labels(200)
        result = kruskal_wallis_per_feature(features, labels)
        assert set(result.data.columns) == {"feature", "p_value", "p_adj"}
        assert result.data["feature"].to_list() == ["separable", "noisy", "constant"]

    def test_separable_feature_is_significant(self) -> None:
        rng = np.random.default_rng(1)
        features = _make_feature_table(rng, n=200)
        labels = _make_labels(200)
        result = kruskal_wallis_per_feature(features, labels)
        p_value = result.data.filter(pl.col("feature") == "separable")["p_value"][0]
        assert p_value < 1e-6

    def test_constant_feature_gets_nan(self) -> None:
        rng = np.random.default_rng(2)
        features = _make_feature_table(rng, n=200)
        labels = _make_labels(200)
        result = kruskal_wallis_per_feature(features, labels)
        row = result.data.filter(pl.col("feature") == "constant")
        assert math.isnan(row["p_value"][0])
        assert math.isnan(row["p_adj"][0])

    def test_adjusted_pvalues_are_monotone_ge_raw(self) -> None:
        rng = np.random.default_rng(3)
        features = _make_feature_table(rng, n=200)
        labels = _make_labels(200)
        result = kruskal_wallis_per_feature(features, labels)
        for p_raw, p_adj in zip(
            result.data["p_value"].to_list(),
            result.data["p_adj"].to_list(),
            strict=False,
        ):
            if math.isnan(p_raw):
                assert math.isnan(p_adj)
            else:
                assert p_adj >= p_raw

    def test_inner_join_with_missing_samples(self) -> None:
        rng = np.random.default_rng(4)
        features = _make_feature_table(rng, n=200)
        # Labels only cover the first 150 samples; the rest are dropped on join.
        labels = _make_labels(150)
        result = kruskal_wallis_per_feature(features, labels)
        assert result.data.shape[0] == 3

    def test_empty_join_raises(self) -> None:
        rng = np.random.default_rng(5)
        features = _make_feature_table(rng, n=10)
        labels_df = pl.DataFrame(
            {
                "sample_id": np.arange(100, 110, dtype=np.uint32),
                "label": ["a"] * 10,
            }
        )
        labels = SampleLabels(data=labels_df)
        with pytest.raises(ValueError, match="inner join"):
            kruskal_wallis_per_feature(features, labels)

    def test_single_group_label_yields_nan(self) -> None:
        rng = np.random.default_rng(6)
        features = _make_feature_table(rng, n=10)
        # All samples have the same label -> only one group per feature.
        labels_df = pl.DataFrame(
            {
                "sample_id": np.arange(10, dtype=np.uint32),
                "label": ["a"] * 10,
            }
        )
        labels = SampleLabels(data=labels_df)
        result = kruskal_wallis_per_feature(features, labels)
        for p in result.data["p_value"].to_list():
            assert math.isnan(p)


class TestIntersectSignificantFeatures:
    """Tests for the ``intersect_significant_features`` node function."""

    @staticmethod
    def _scores(
        feature_padj: dict[str, float],
    ) -> FeatureSignificance:
        df = pl.DataFrame(
            {
                "feature": list(feature_padj),
                "p_value": list(feature_padj.values()),
                "p_adj": list(feature_padj.values()),
            }
        )
        return FeatureSignificance(data=df)

    def test_returns_sorted_intersection(self) -> None:
        a = self._scores({"x": 0.01, "y": 0.04, "z": 0.2})
        b = self._scores({"x": 0.001, "y": 0.5, "w": 0.02})
        result = intersect_significant_features(0.05, a, b)
        assert result == ["x"]

    def test_threshold_one_returns_all_shared(self) -> None:
        a = self._scores({"x": 0.5, "y": 0.9})
        b = self._scores({"x": 0.6, "y": 0.7})
        result = intersect_significant_features(1.0, a, b)
        assert result == ["x", "y"]

    def test_threshold_zero_returns_empty(self) -> None:
        a = self._scores({"x": 0.5})
        b = self._scores({"x": 0.5})
        result = intersect_significant_features(0.0, a, b)
        assert result == []

    def test_no_partitions_returns_empty(self) -> None:
        assert intersect_significant_features(0.05) == []

    def test_intersection_is_empty_when_disjoint(self) -> None:
        a = self._scores({"x": 0.01})
        b = self._scores({"y": 0.01})
        result = intersect_significant_features(0.05, a, b)
        assert result == []
