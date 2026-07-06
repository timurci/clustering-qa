"""Nodes for the clustering agreement pipeline."""

from functools import reduce

import numpy as np
import polars as pl
from scipy.stats import false_discovery_control, kruskal

from clustering_qa.datasets.feature_significance import FeatureSignificance
from clustering_qa.datasets.feature_table import FeatureTable, SampleLabels

_MIN_GROUPS_FOR_KRUSKAL = 2


def kruskal_wallis_per_feature(
    features: FeatureTable, labels: SampleLabels
) -> FeatureSignificance:
    """Run a Kruskal-Wallis H-test for every feature with ``label`` as the strata.

    P-values are adjusted across features using the Benjamini-Hochberg
    procedure. Features that cannot be tested (single group, constant
    values, or that degenerate to fewer than two groups after the inner
    join) receive ``NaN`` p-values and ``NaN`` adjusted p-values.

    Args:
        features: Feature table whose non-id numeric columns are tested.
        labels: Per-sample labels that define the strata for the test.

    Returns:
        A :class:`FeatureSignificance` with columns ``feature``, ``p_value``,
        and ``p_adj``.
    """
    joined = features.data.join(labels.data, on="sample_id", how="inner")
    if joined.is_empty():
        msg = "inner join of features and labels produced no rows"
        raise ValueError(msg)

    feature_columns = [col for col in features.data.columns if col != "sample_id"]
    p_values = np.array([_kruskal_pvalue(joined, col) for col in feature_columns])

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


def _kruskal_pvalue(joined: pl.DataFrame, feature_col: str) -> float:
    """Compute a Kruskal-Wallis p-value for ``feature_col``.

    Returns ``NaN`` when the feature cannot be tested.
    """
    groups = [
        df[feature_col].to_numpy()
        for df in joined.partition_by("label", maintain_order=False)
    ]
    if len(groups) < _MIN_GROUPS_FOR_KRUSKAL:
        return float("nan")
    result = kruskal(*groups)
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
