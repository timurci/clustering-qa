"""Tests for the feature table dataset module."""

from pathlib import Path

import polars as pl
import pytest

from clustering_qa.datasets.feature_table import (
    FeatureTable,
    FeatureTableDataset,
    SampleLabels,
    SampleLabelsDataset,
)


def _make_feature_table() -> FeatureTable:
    return FeatureTable(
        data=pl.DataFrame(
            {
                "sample_id": ["a", "b", "c"],
                "x": [1.0, 2.0, 3.0],
                "y": [4.0, 5.0, 6.0],
            }
        )
    )


def _make_labels() -> SampleLabels:
    return SampleLabels(
        data=pl.DataFrame(
            {
                "sample_id": ["a", "b", "c"],
                "label": ["0", "1", "0"],
            }
        )
    )


class TestFeatureTable:
    """Post-init validation for the :class:`FeatureTable` dataclass."""

    def test_missing_sample_id_column_raises(self) -> None:
        bad = pl.DataFrame({"x": [1.0]})
        with pytest.raises(AssertionError, match="sample_id"):
            FeatureTable(data=bad)


class TestSampleLabels:
    """Post-init validation for the :class:`SampleLabels` dataclass."""

    def test_missing_label_column_raises(self) -> None:
        bad = pl.DataFrame({"sample_id": ["a"]})
        with pytest.raises(AssertionError, match="label"):
            SampleLabels(data=bad)

    def test_label_is_cast_to_categorical(self) -> None:
        labels = SampleLabels(
            data=pl.DataFrame({"sample_id": ["a", "b"], "label": ["0", "1"]})
        )
        assert labels.data.schema["label"] == pl.Categorical


class TestFeatureTableDataset:
    """Round-trip tests for :class:`FeatureTableDataset`."""

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        filepath = tmp_path / "features.parquet"
        dataset = FeatureTableDataset(str(filepath))
        original = _make_feature_table()
        dataset.save(original)
        loaded = dataset.load()
        assert loaded.data.columns == ["sample_id", "x", "y"]
        assert loaded.data["sample_id"].to_list() == ["a", "b", "c"]

    def test_describe(self, tmp_path: Path) -> None:
        filepath = str(tmp_path / "features.parquet")
        dataset = FeatureTableDataset(filepath)
        assert dataset._describe() == {"filepath": filepath}


class TestSampleLabelsDataset:
    """Round-trip tests for :class:`SampleLabelsDataset`."""

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        filepath = tmp_path / "labels.csv"
        dataset = SampleLabelsDataset(str(filepath))
        original = _make_labels()
        dataset.save(original)
        loaded = dataset.load()
        assert loaded.data.columns == ["sample_id", "label"]
        assert loaded.data["sample_id"].to_list() == ["a", "b", "c"]

    def test_describe(self, tmp_path: Path) -> None:
        filepath = str(tmp_path / "labels.csv")
        dataset = SampleLabelsDataset(filepath)
        assert dataset._describe() == {"filepath": filepath}
