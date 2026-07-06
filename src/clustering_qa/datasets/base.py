"""Base dataset definitions."""

from dataclasses import dataclass
from typing import ClassVar

import polars as pl


@dataclass(frozen=True)
class AnnotatedDataFrame:
    """Base class for datasets backed by a Polars DataFrame with column validation."""

    data: pl.DataFrame
    required_columns: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self) -> None:
        """Validate that all required columns are present."""
        missing = [c for c in self.required_columns if c not in self.data.columns]
        if missing:
            error_msg = (
                f"{self.__class__.__name__} is missing {', '.join(missing)} column(s)"
            )
            raise AssertionError(error_msg)
