"""Dataset definitions for embedding."""

from dataclasses import dataclass

import polars as pl
from kedro.io import AbstractDataset


@dataclass(frozen=True)
class Embeddings:
    """Represents an embedded dataset."""

    data: pl.DataFrame

    def __post_init__(self) -> None:
        """Post-initialization checks for Embeddings."""
        _assert_required_cols(self.data, ["sample_id"])


@dataclass(frozen=True)
class EmbeddingLabels:
    """Represents the labels for an embedded dataset."""

    data: pl.DataFrame

    def __post_init__(self) -> None:
        """Post-initialization checks for EmbeddingLabels."""
        _assert_required_cols(self.data, ["sample_id", "label"])

        # Ensure label column is categorical
        if self.data.schema["label"] != pl.Categorical:
            object.__setattr__(
                self,
                "data",
                self.data.with_columns(pl.col("label").cast(pl.Categorical)),
            )


def _assert_required_cols(df: pl.DataFrame, required_cols: list[str]) -> None:
    missing_cols = [x for x in required_cols if x not in df.columns]
    if missing_cols:
        error_msg = f"dataset is missing {', '.join(missing_cols)} column(s)"
        raise AssertionError(error_msg)


class EmbeddingsDataset(AbstractDataset[Embeddings, Embeddings]):
    """Loader for Embeddings."""

    def __init__(self, filepath: str) -> None:
        """Initializes the EmbeddingsDataset with the given filepath."""
        self._filepath = filepath

    def load(self) -> Embeddings:
        """Loads the embedded dataset from the given filepath."""
        df = pl.read_parquet(self._filepath)
        return Embeddings(data=df)

    def save(self, data: Embeddings) -> None:
        """Saves the embedded dataset to the given filepath."""
        data.data.write_parquet(self._filepath, compression="zstd", compression_level=5)


class EmbeddingLabelsDataset(AbstractDataset):
    """Loader for Embedding Labels."""

    def __init__(self, filepath: str) -> None:
        """Initializes the EmbeddingLabelsDataset with the given filepath."""
        self._filepath = filepath

    def load(self) -> EmbeddingLabels:
        """Loads the embedding labels from the given filepath."""
        df = pl.read_parquet(self._filepath)
        return EmbeddingLabels(data=df)

    def save(self, data: pl.DataFrame) -> None:
        """Saves the embedding labels to the given filepath."""
        data.write_parquet(self._filepath, compression="zstd", compression_level=5)
