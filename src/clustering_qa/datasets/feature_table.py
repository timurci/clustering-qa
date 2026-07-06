"""Dataset definitions for feature tables and sample labels."""

from dataclasses import dataclass
from typing import Any, ClassVar

import polars as pl
from kedro.io import AbstractDataset, DatasetError

from clustering_qa.datasets.base import AnnotatedDataFrame


class IdColumnNotFoundError(Exception):
    """Raised when the configured id column is not found in the loaded data."""

    def __init__(self, id_column: str, available_columns: list[str]) -> None:
        """Store the missing column and the columns that were available."""
        self.id_column = id_column
        self.available_columns = available_columns
        super().__init__(
            f"id_column '{id_column}' not found; available columns: {available_columns}"
        )


@dataclass(frozen=True)
class FeatureTable(AnnotatedDataFrame):
    """Represents a sample-by-feature table."""

    required_columns: ClassVar[tuple[str, ...]] = ("sample_id",)


@dataclass(frozen=True)
class SampleLabels(AnnotatedDataFrame):
    """Represents the labels assigned to samples."""

    required_columns: ClassVar[tuple[str, ...]] = ("sample_id", "label")

    def __post_init__(self) -> None:
        """Post-initialization checks for SampleLabels."""
        super().__post_init__()

        # Ensure label column is categorical
        if self.data.schema["label"] != pl.Categorical:
            object.__setattr__(
                self,
                "data",
                self.data.with_columns(
                    pl.col("label").cast(pl.Utf8).cast(pl.Categorical)
                ),
            )


class FeatureTableDataset(AbstractDataset[FeatureTable, FeatureTable]):
    """Loader for FeatureTable."""

    def __init__(self, filepath: str, id_column: str) -> None:
        """Initializes the FeatureTableDataset with the given filepath and id column."""
        self._filepath = filepath
        self._id_column = id_column

    def load(self) -> FeatureTable:
        """Loads the feature table from the given filepath."""
        df = pl.read_parquet(self._filepath)
        if self._id_column not in df.columns:
            exc = IdColumnNotFoundError(self._id_column, df.columns)
            raise DatasetError(str(exc)) from exc
        df = df.rename({self._id_column: "sample_id"})
        return FeatureTable(data=df)

    def save(self, data: FeatureTable) -> None:
        """Saves the feature table to the given filepath."""
        data.data.write_parquet(self._filepath, compression="zstd", compression_level=5)

    def _describe(self) -> dict[str, Any]:
        return {"filepath": self._filepath, "id_column": self._id_column}


class SampleLabelsDataset(AbstractDataset[SampleLabels, SampleLabels]):
    """Loader for SampleLabels."""

    def __init__(self, filepath: str, id_column: str) -> None:
        """Initializes the SampleLabelsDataset with the given filepath and id column."""
        self._filepath = filepath
        self._id_column = id_column

    def load(self) -> SampleLabels:
        """Loads the sample labels from the given filepath."""
        df = pl.read_csv(self._filepath)
        if self._id_column not in df.columns:
            exc = IdColumnNotFoundError(self._id_column, df.columns)
            raise DatasetError(str(exc)) from exc
        df = df.rename({self._id_column: "sample_id"})
        return SampleLabels(data=df)

    def save(self, data: SampleLabels) -> None:
        """Saves the sample labels to the given filepath."""
        data.data.write_csv(self._filepath)

    def _describe(self) -> dict[str, Any]:
        return {"filepath": self._filepath, "id_column": self._id_column}
