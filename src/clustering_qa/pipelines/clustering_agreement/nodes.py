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
    join) receive ``NaN`` p-values, ``NaN`` adjusted p-values, and ``NaN``
    eta-squared.

    Args:
        features: Feature table whose non-id numeric columns are tested.
        labels: Per-sample labels that define the strata for the test.
        n_features: Optional limit on the number of features tested; the
            first ``n_features`` columns are used. ``None`` tests all
            features.

    Returns:
        A :class:`FeatureSignificance` with columns ``feature``, ``p_value``,
        ``p_adj``, and ``eta_squared`` (the Kruskal-Wallis effect size
        ``(H - k + 1) / (n - k)``, clamped at 0).
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
    n_groups = len(group_matrices)
    if n_groups < _MIN_GROUPS_FOR_KRUSKAL:
        return FeatureSignificance(
            data=pl.DataFrame(
                {
                    "feature": feature_columns,
                    "p_value": np.full(len(feature_columns), np.nan),
                    "p_adj": np.full(len(feature_columns), np.nan),
                    "eta_squared": np.full(len(feature_columns), np.nan),
                }
            )
        )
    tested = [_kruskal_test(group_matrices, i) for i in range(len(feature_columns))]
    p_values = np.array([p_value for p_value, _ in tested])
    h_statistics = np.array([h_statistic for _, h_statistic in tested])
    eta_squared = np.maximum(
        0.0, (h_statistics - n_groups + 1) / (joined.height - n_groups)
    )

    return FeatureSignificance(
        data=pl.DataFrame(
            {
                "feature": feature_columns,
                "p_value": p_values,
                "p_adj": _adjust_pvalues(p_values),
                "eta_squared": eta_squared,
            }
        )
    )


def select_consensus_features(
    p_adj_threshold: float, n_selected: int, *partition_scores: FeatureSignificance
) -> list[str]:
    """Select per-partition top features and return their intersection.

    Per partition, features with ``p_adj < p_adj_threshold`` are ranked by
    ``eta_squared`` (descending) and the first ``n_selected`` are kept;
    partitions with fewer survivors keep all of them. The consensus is the
    sorted intersection of the per-partition selections and is used as-is
    by downstream analysis.

    Args:
        p_adj_threshold: Maximum adjusted p-value that still counts as
            significant.
        n_selected: Maximum number of features selected per partition.
        partition_scores: One :class:`FeatureSignificance` per partition.

    Returns:
        Sorted list of feature names selected in every partition.
    """
    if n_selected <= 0:
        msg = f"n_selected must be positive, got {n_selected}"
        raise ValueError(msg)
    if not partition_scores:
        return []

    selected_sets = [
        _select_top_features(scores, p_adj_threshold, n_selected)
        for scores in partition_scores
    ]
    intersection = reduce(set.intersection, selected_sets)
    return sorted(intersection)


def _select_top_features(
    scores: FeatureSignificance, p_adj_threshold: float, n_selected: int
) -> set[str]:
    """Return the top ``n_selected`` features of one partition's scores."""
    survivors = scores.data.filter(
        (pl.col("p_adj") < p_adj_threshold) & pl.col("eta_squared").is_finite()
    )
    ranked = survivors.sort(["eta_squared", "feature"], descending=[True, False])[
        "feature"
    ].to_list()
    return set(ranked[:n_selected])


def _kruskal_test(
    group_matrices: list[np.ndarray], feature_index: int
) -> tuple[float, float]:
    """Run a Kruskal-Wallis test for ``feature_index``.

    The ``group_matrices`` are the per-label partitions of the joined
    feature table, with one row per sample and one column per feature.

    Returns ``(p_value, h_statistic)``; ``(NaN, NaN)`` when the feature
    cannot be tested. Non-finite results are untestable: for constant
    input scipy's tie correction divides an otherwise-zero statistic by
    zero, and the floating-point residue of the division decides between
    NaN and an infinite statistic carrying a spurious p-value.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = kruskal(*[group[:, feature_index] for group in group_matrices])
    if not (np.isfinite(result.statistic) and np.isfinite(result.pvalue)):
        return float("nan"), float("nan")
    return float(result.pvalue), float(result.statistic)


def _adjust_pvalues(p_values: np.ndarray) -> np.ndarray:
    """Apply Benjamini-Hochberg adjustment, preserving ``NaN`` entries."""
    nan_mask = np.isnan(p_values)
    adjusted = np.full_like(p_values, np.nan, dtype=float)
    if (~nan_mask).any():
        adjusted[~nan_mask] = false_discovery_control(p_values[~nan_mask], method="bh")
    return adjusted
