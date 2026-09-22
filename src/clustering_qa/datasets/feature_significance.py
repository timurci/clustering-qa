"""Dataset definitions for feature significance scores."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import polars as pl
from kedro.io import AbstractDataset

from clustering_qa.datasets.base import AnnotatedDataFrame

# Intentionally limited to 3 decimals: reporting granularity finer than
# "<0.001" is not used downstream, and full precision inflates the CSVs
# without adding information.
CSV_FLOAT_PRECISION = 3


@dataclass(frozen=True)
class FeatureSignificance(AnnotatedDataFrame):
    """Per-feature significance scores and effect sizes from a Kruskal-Wallis test."""

    required_columns: ClassVar[tuple[str, ...]] = (
        "feature",
        "p_value",
        "p_adj",
        "eta_squared",
    )


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
        Path(self._filepath).parent.mkdir(parents=True, exist_ok=True)
        data.data.write_csv(self._filepath, float_precision=CSV_FLOAT_PRECISION)

    def _describe(self) -> dict[str, Any]:
        return {"filepath": self._filepath}
