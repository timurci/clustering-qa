"""Unit tests for the clustering_agreement pipeline nodes."""

import math
import warnings

import numpy as np
import polars as pl
import pytest

from clustering_qa.datasets.feature_significance import FeatureSignificance
from clustering_qa.datasets.feature_table import FeatureTable, SampleLabels
from clustering_qa.pipelines.clustering_agreement.nodes import (
    kruskal_wallis_per_feature,
    select_consensus_features,
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
        assert set(result.data.columns) == {
            "feature",
            "p_value",
            "p_adj",
            "eta_squared",
        }
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
        assert math.isnan(row["eta_squared"][0])

    def test_eta_squared_orders_effects_and_is_bounded(self) -> None:
        rng = np.random.default_rng(2)
        features = _make_feature_table(rng, n=200)
        labels = _make_labels(200)
        result = kruskal_wallis_per_feature(features, labels)
        eta = {row["feature"]: row["eta_squared"] for row in result.data.to_dicts()}
        assert eta["separable"] > eta["noisy"]
        for value in eta.values():
            if not math.isnan(value):
                assert 0.0 <= value <= 1.0

    def test_constant_feature_is_nan_regardless_of_tie_rounding(self) -> None:
        """Regression: constant features must be NaN for any group sizes.

        scipy's tie correction divides h by a tie factor of exactly 0 for
        constant input; the floating-point residue of the otherwise-zero h
        then decides between NaN, +inf (p=0.0), and -inf (p=1.0). With
        group sizes 25/25/31/24 (n=105) this used to yield p=0.0, which
        passed the NaN check and entered the significant set.
        """
        rng = np.random.default_rng(12)
        n = 105
        features = FeatureTable(
            data=pl.DataFrame(
                {
                    "sample_id": np.arange(n, dtype=np.uint32),
                    "signal": rng.normal(size=n),
                    "constant": np.zeros(n),
                }
            )
        )
        labels = SampleLabels(
            data=pl.DataFrame(
                {
                    "sample_id": np.arange(n, dtype=np.uint32),
                    "label": ["a"] * 25 + ["b"] * 25 + ["c"] * 31 + ["d"] * 24,
                }
            )
        )
        result = kruskal_wallis_per_feature(features, labels)
        row = result.data.filter(pl.col("feature") == "constant")
        assert math.isnan(row["p_value"][0])
        assert math.isnan(row["eta_squared"][0])

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

    def test_n_features_limits_to_first_columns(self) -> None:
        rng = np.random.default_rng(7)
        features = _make_feature_table(rng, n=200)
        labels = _make_labels(200)
        result = kruskal_wallis_per_feature(features, labels, n_features=2)
        assert result.data["feature"].to_list() == ["separable", "noisy"]

    def test_n_features_none_tests_all(self) -> None:
        rng = np.random.default_rng(8)
        features = _make_feature_table(rng, n=200)
        labels = _make_labels(200)
        result = kruskal_wallis_per_feature(features, labels, n_features=None)
        assert result.data["feature"].to_list() == ["separable", "noisy", "constant"]

    def test_n_features_warns_when_truncating(self) -> None:
        rng = np.random.default_rng(9)
        features = _make_feature_table(rng, n=200)
        labels = _make_labels(200)
        with pytest.warns(UserWarning, match="n_features=2 limits testing"):
            result = kruskal_wallis_per_feature(features, labels, n_features=2)
        assert result.data.shape[0] == 2

    def test_n_features_larger_than_total_does_not_warn(self) -> None:
        rng = np.random.default_rng(10)
        features = _make_feature_table(rng, n=200)
        labels = _make_labels(200)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = kruskal_wallis_per_feature(features, labels, n_features=10)
        assert result.data["feature"].to_list() == ["separable", "noisy", "constant"]
        assert not any("n_features" in str(w.message) for w in caught)

    def test_negative_n_features_raises(self) -> None:
        rng = np.random.default_rng(11)
        features = _make_feature_table(rng, n=200)
        labels = _make_labels(200)
        with pytest.raises(ValueError, match="n_features must be positive"):
            kruskal_wallis_per_feature(features, labels, n_features=-1)

    def test_zero_n_features_raises(self) -> None:
        rng = np.random.default_rng(11)
        features = _make_feature_table(rng, n=200)
        labels = _make_labels(200)
        with pytest.raises(ValueError, match="n_features must be positive"):
            kruskal_wallis_per_feature(features, labels, n_features=0)


class TestSelectConsensusFeatures:
    """Tests for the ``select_consensus_features`` node function."""

    @staticmethod
    def _scores(rows: dict[str, tuple[float, float]]) -> FeatureSignificance:
        """Builds scores from ``feature -> (p_adj, eta_squared)``."""
        return FeatureSignificance(
            data=pl.DataFrame(
                {
                    "feature": list(rows),
                    "p_value": [p for p, _ in rows.values()],
                    "p_adj": [p for p, _ in rows.values()],
                    "eta_squared": [e for _, e in rows.values()],
                }
            )
        )

    def test_selects_top_eta_per_partition_and_intersects(self) -> None:
        a = self._scores({"x": (0.01, 0.5), "y": (0.01, 0.4), "z": (0.01, 0.1)})
        b = self._scores({"x": (0.01, 0.3), "y": (0.01, 0.2), "w": (0.01, 0.9)})
        # Top-2 per partition: a -> {x, y}; b -> {w, x}.
        assert select_consensus_features(0.05, 2, a, b) == ["x"]

    def test_fdr_gate_excludes_nonsignificant(self) -> None:
        a = self._scores({"x": (0.01, 0.5), "y": (0.2, 0.9)})
        b = self._scores({"x": (0.01, 0.5), "y": (0.2, 0.9)})
        assert select_consensus_features(0.05, 10, a, b) == ["x"]

    def test_underfill_keeps_all_survivors(self) -> None:
        a = self._scores({"x": (0.01, 0.5), "y": (0.02, 0.4)})
        b = self._scores({"x": (0.01, 0.3), "y": (0.02, 0.2)})
        assert select_consensus_features(0.05, 2000, a, b) == ["x", "y"]

    def test_threshold_zero_returns_empty(self) -> None:
        a = self._scores({"x": (0.5, 0.5)})
        b = self._scores({"x": (0.5, 0.5)})
        assert select_consensus_features(0.0, 10, a, b) == []

    def test_no_partitions_returns_empty(self) -> None:
        assert select_consensus_features(0.05, 10) == []

    def test_intersection_is_empty_when_disjoint(self) -> None:
        a = self._scores({"x": (0.01, 0.5)})
        b = self._scores({"y": (0.01, 0.5)})
        assert select_consensus_features(0.05, 10, a, b) == []

    def test_negative_n_selected_raises(self) -> None:
        a = self._scores({"x": (0.01, 0.5)})
        with pytest.raises(ValueError, match="n_selected must be positive"):
            select_consensus_features(0.05, -1, a)

    def test_zero_n_selected_raises(self) -> None:
        a = self._scores({"x": (0.01, 0.5)})
        with pytest.raises(ValueError, match="n_selected must be positive"):
            select_consensus_features(0.05, 0, a)
