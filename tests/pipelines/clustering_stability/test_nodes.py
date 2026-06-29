"""Unit tests for the clustering_stability pipeline nodes."""

import numpy as np
import polars as pl
import pytest

from clustering_qa.datasets.feature_table import FeatureTable, SampleLabels
from clustering_qa.pipelines.clustering_stability.nodes import (
    assess_stability,
    kmeans_clustering,
)


def _make_feature_table(n: int = 10, seed: int = 0) -> FeatureTable:
    rng = np.random.default_rng(seed)
    return FeatureTable(
        data=pl.DataFrame(
            {
                "sample_id": [f"s{i}" for i in range(n)],
                "x": rng.normal(size=n),
                "y": rng.normal(size=n),
            }
        )
    )


class TestKMeansClustering:
    """Tests for the ``kmeans_clustering`` node function."""

    def test_returns_expected_columns(self) -> None:
        features = _make_feature_table(n=10)
        labels = kmeans_clustering(features, n_clusters=2, seed=0)
        assert labels.data.columns == ["sample_id", "label"]
        assert (
            labels.data["sample_id"].to_list() == features.data["sample_id"].to_list()
        )
        assert labels.data.schema["label"] == pl.Categorical

    def test_number_of_labels_matches_samples(self) -> None:
        features = _make_feature_table(n=20)
        labels = kmeans_clustering(features, n_clusters=3, seed=0)
        assert labels.data.shape[0] == 20


class TestAssessStability:
    """Tests for the ``assess_stability`` node function."""

    def test_identical_trials_have_perfect_scores(self) -> None:
        labels = SampleLabels(
            data=pl.DataFrame(
                {
                    "sample_id": ["a", "b", "c", "d"],
                    "label": pl.Series(["0", "1", "0", "1"], dtype=pl.Categorical),
                }
            )
        )
        scores = assess_stability(labels, labels, labels)
        assert scores["ari"] == pytest.approx(1.0)
        assert scores["nmi"] == pytest.approx(1.0)

    def test_different_trials_have_lower_scores(self) -> None:
        trial1 = SampleLabels(
            data=pl.DataFrame(
                {
                    "sample_id": ["a", "b", "c", "d"],
                    "label": pl.Series(["0", "0", "1", "1"], dtype=pl.Categorical),
                }
            )
        )
        trial2 = SampleLabels(
            data=pl.DataFrame(
                {
                    "sample_id": ["a", "b", "c", "d"],
                    "label": pl.Series(["0", "1", "0", "1"], dtype=pl.Categorical),
                }
            )
        )
        scores = assess_stability(trial1, trial2)
        assert scores["ari"] < 1.0
        assert scores["nmi"] < 1.0
