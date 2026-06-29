"""Tests for the FeatureSignificance dataset module."""

from pathlib import Path

import polars as pl
import pytest

from clustering_qa.datasets.feature_significance import (
    FeatureSignificance,
    FeatureSignificanceDataset,
)


def _make_scores() -> FeatureSignificance:
    return FeatureSignificance(
        data=pl.DataFrame(
            {
                "feature": ["a", "b", "c"],
                "p_value": [0.001, 0.04, 0.6],
                "p_adj": [0.003, 0.12, 0.6],
            }
        )
    )


class TestFeatureSignificance:
    """Post-init validation for the :class:`FeatureSignificance` dataclass."""

    def test_missing_required_column_raises(self) -> None:
        bad = pl.DataFrame({"feature": ["x"], "p_value": [0.5]})
        with pytest.raises(AssertionError, match="p_adj"):
            FeatureSignificance(data=bad)


class TestFeatureSignificanceDataset:
    """Round-trip tests for :class:`FeatureSignificanceDataset`."""

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        filepath = tmp_path / "scores.csv"
        dataset = FeatureSignificanceDataset(str(filepath))
        original = _make_scores()
        dataset.save(original)
        loaded = dataset.load()
        assert loaded.data.columns == ["feature", "p_value", "p_adj"]
        assert loaded.data["feature"].to_list() == ["a", "b", "c"]
        assert loaded.data["p_value"].to_list() == [0.001, 0.04, 0.6]
        assert loaded.data["p_adj"].to_list() == [0.003, 0.12, 0.6]

    def test_describe(self, tmp_path: Path) -> None:
        filepath = str(tmp_path / "scores.csv")
        dataset = FeatureSignificanceDataset(filepath)
        assert dataset._describe() == {"filepath": filepath}
