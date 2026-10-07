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
                "min_gap": [1.2, 0.4, -0.1],
                "max_gap": [2.0, 0.9, 0.05],
            }
        )
    )


class TestFeatureSignificance:
    """Post-init validation for the :class:`FeatureSignificance` dataclass."""

    def test_missing_required_column_raises(self) -> None:
        bad = pl.DataFrame({"feature": ["x"], "p_value": [0.5], "p_adj": [0.5]})
        with pytest.raises(AssertionError, match="min_gap"):
            FeatureSignificance(data=bad)


class TestFeatureSignificanceDataset:
    """Round-trip tests for the :class:`FeatureSignificanceDataset`."""

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        filepath = tmp_path / "scores.csv"
        dataset = FeatureSignificanceDataset(str(filepath))
        original = _make_scores()
        dataset.save(original)
        loaded = dataset.load()
        assert loaded.data.columns == [
            "feature",
            "p_value",
            "p_adj",
            "min_gap",
            "max_gap",
        ]
        assert loaded.data["feature"].to_list() == ["a", "b", "c"]
        assert loaded.data["p_value"].to_list() == [0.001, 0.04, 0.6]
        assert loaded.data["p_adj"].to_list() == [0.003, 0.12, 0.6]
        assert loaded.data["min_gap"].to_list() == [1.2, 0.4, -0.1]
        assert loaded.data["max_gap"].to_list() == [2.0, 0.9, 0.05]

    def test_save_creates_parent_directories(self, tmp_path: Path) -> None:
        filepath = tmp_path / "nested" / "dir" / "scores.csv"
        dataset = FeatureSignificanceDataset(str(filepath))
        dataset.save(_make_scores())
        loaded = dataset.load()
        assert loaded.data.columns == [
            "feature",
            "p_value",
            "p_adj",
            "min_gap",
            "max_gap",
        ]

    def test_save_limits_float_precision(self, tmp_path: Path) -> None:
        filepath = tmp_path / "scores.csv"
        dataset = FeatureSignificanceDataset(str(filepath))
        original = FeatureSignificance(
            data=pl.DataFrame(
                {
                    "feature": ["a", "b"],
                    "p_value": [0.00123456789, 0.000987654321],
                    "p_adj": [0.00543210987, 0.000043210987],
                    "min_gap": [0.123456789, 0.000987654321],
                    "max_gap": [1.987654321, 0.000111222333],
                }
            )
        )
        dataset.save(original)
        loaded = dataset.load()
        assert loaded.data["p_value"].to_list() == [
            round(0.00123456789, 3),
            round(0.000987654321, 3),
        ]
        assert loaded.data["p_adj"].to_list() == [
            round(0.00543210987, 3),
            round(0.000043210987, 3),
        ]
        assert loaded.data["min_gap"].to_list() == [
            round(0.123456789, 3),
            round(0.000987654321, 3),
        ]
        assert loaded.data["max_gap"].to_list() == [
            round(1.987654321, 3),
            round(0.000111222333, 3),
        ]

    def test_describe(self, tmp_path: Path) -> None:
        filepath = str(tmp_path / "scores.csv")
        dataset = FeatureSignificanceDataset(filepath)
        assert dataset._describe() == {"filepath": filepath}
