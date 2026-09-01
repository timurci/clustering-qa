"""Nodes for the clustering agreement pipeline."""

import warnings
from functools import reduce

import numpy as np
import polars as pl
from scipy.stats import false_discovery_control, kruskal

from clustering_qa.datasets.feature_significance import FeatureSignificance
from clustering_qa.datasets.feature_table import FeatureTable, SampleLabels

_MIN_GROUPS_FOR_KRUSKAL = 2


def kruskal_wallis_per_feature(
    features: FeatureTable, labels: SampleLabels, n_features: int | None = None
) -> FeatureSignificance:
    """Run a Kruskal-Wallis H-test for every feature with ``label`` as the strata.

    P-values are adjusted across features using the Benjamini-Hochberg
    procedure. Features that cannot be tested (single group, constant
    values, or that degenerate to fewer than two groups after the inner
    join) receive ``NaN`` p-values and ``NaN`` adjusted p-values.

    Args:
        features: Feature table whose non-id numeric columns are tested.
        labels: Per-sample labels that define the strata for the test.
        n_features: Optional limit on the number of features tested; the
            first ``n_features`` columns are used. ``None`` tests all
            features.

    Returns:
        A :class:`FeatureSignificance` with columns ``feature``, ``p_value``,
        and ``p_adj``.
    """
    if n_features is not None and n_features <= 0:
        msg = f"n_features must be positive, got {n_features}"
        raise ValueError(msg)

    joined = features.data.join(labels.data, on="sample_id", how="inner")
    if joined.is_empty():
        msg = "inner join of features and labels produced no rows"
        raise ValueError(msg)

    feature_columns = [col for col in features.data.columns if col != "sample_id"]
    if n_features is not None and len(feature_columns) > n_features:
        truncated = feature_columns[:n_features]
        warnings.warn(
            f"n_features={n_features} limits testing to the first "
            f"{len(truncated)} of {len(feature_columns)} features",
            stacklevel=2,
        )
        feature_columns = truncated

    group_matrices = [
        partition.select(feature_columns).cast(pl.Float64).to_numpy()
        for partition in joined.partition_by("label", maintain_order=True)
    ]
    if len(group_matrices) < _MIN_GROUPS_FOR_KRUSKAL:
        return FeatureSignificance(
            data=pl.DataFrame(
                {
                    "feature": feature_columns,
                    "p_value": np.full(len(feature_columns), np.nan),
                    "p_adj": np.full(len(feature_columns), np.nan),
                }
            )
        )
    p_values = np.array(
        [_kruskal_pvalue(group_matrices, i) for i in range(len(feature_columns))]
    )

    return FeatureSignificance(
        data=pl.DataFrame(
            {
                "feature": feature_columns,
                "p_value": p_values,
                "p_adj": _adjust_pvalues(p_values),
            }
        )
    )


def intersect_significant_features(
    p_adj_threshold: float, *partition_scores: FeatureSignificance
) -> list[str]:
    """Return the intersection of features significant in every partition.

    Significance is defined as ``p_adj < p_adj_threshold``.

    Args:
        p_adj_threshold: Maximum adjusted p-value that still counts as
            significant.
        partition_scores: One :class:`FeatureSignificance` per partition.

    Returns:
        Sorted list of feature names significant in every partition.
    """
    if not partition_scores:
        return []

    significant_sets = [
        set(scores.data.filter(pl.col("p_adj") < p_adj_threshold)["feature"].to_list())
        for scores in partition_scores
    ]
    intersection = reduce(set.intersection, significant_sets)
    return sorted(intersection)


def _kruskal_pvalue(group_matrices: list[np.ndarray], feature_index: int) -> float:
    """Compute a Kruskal-Wallis p-value for ``feature_index``.

    The ``group_matrices`` are the per-label partitions of the joined
    feature table, with one row per sample and one column per feature.

    Returns ``NaN`` when the feature cannot be tested.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = kruskal(*[group[:, feature_index] for group in group_matrices])
    if result.pvalue is None or np.isnan(result.pvalue):
        return float("nan")
    return float(result.pvalue)


def _adjust_pvalues(p_values: np.ndarray) -> np.ndarray:
    """Apply Benjamini-Hochberg adjustment, preserving ``NaN`` entries."""
    nan_mask = np.isnan(p_values)
    adjusted = np.full_like(p_values, np.nan, dtype=float)
    if (~nan_mask).any():
        adjusted[~nan_mask] = false_discovery_control(p_values[~nan_mask], method="bh")
    return adjusted
