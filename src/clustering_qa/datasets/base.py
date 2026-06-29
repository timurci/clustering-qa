"""Base dataset definitions."""

from dataclasses import dataclass, field
from typing import ClassVar

import polars as pl


@dataclass(frozen=True)
class AnnotatedDataFrame:
    """Base class for datasets backed by a Polars DataFrame.

    Subclasses declare their required columns via ``required_columns``.
    The ``id_column`` identifies rows and defaults to ``"sample_id"``.

    Args:
        data: The Polars DataFrame backing this dataset.
        id_column: Name of the column that identifies rows.
    """

    data: pl.DataFrame
    id_column: str = field(default="sample_id", repr=True)
    required_columns: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self) -> None:
        """Validate that all required columns are present."""
        missing = [c for c in self.required_columns if c not in self.data.columns]
        if missing:
            error_msg = (
                f"{self.__class__.__name__} is missing {', '.join(missing)} column(s)"
            )
            raise AssertionError(error_msg)
