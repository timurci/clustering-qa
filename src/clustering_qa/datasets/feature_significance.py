"""Dataset definitions for feature significance scores."""

from dataclasses import dataclass, field
from typing import Any, ClassVar

import polars as pl
from kedro.io import AbstractDataset

from clustering_qa.datasets.base import AnnotatedDataFrame


@dataclass(frozen=True)
class FeatureSignificance(AnnotatedDataFrame):
    """Represents per-feature significance scores from a Kruskal-Wallis test."""

    id_column: str = field(default="feature")
    required_columns: ClassVar[tuple[str, ...]] = ("feature", "p_value", "p_adj")


class FeatureSignificanceDataset(
    AbstractDataset[FeatureSignificance, FeatureSignificance]
):
    """Loader for FeatureSignificance."""

    def __init__(self, filepath: str) -> None:
        """Initializes the FeatureSignificanceDataset with the given filepath."""
        self._filepath = filepath

    def load(self) -> FeatureSignificance:
        """Loads the feature significance scores from the given filepath."""
        df = pl.read_csv(self._filepath)
        return FeatureSignificance(data=df)

    def save(self, data: FeatureSignificance) -> None:
        """Saves the feature significance scores to the given filepath."""
        data.data.write_csv(self._filepath)

    def _describe(self) -> dict[str, Any]:
        return {"filepath": self._filepath}
