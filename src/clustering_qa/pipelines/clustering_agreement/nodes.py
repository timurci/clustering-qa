"""Nodes for the clustering agreement pipeline."""

import warnings
from functools import reduce

import numpy as np
import polars as pl
from scipy.stats import false_discovery_control, kruskal

from clustering_qa.datasets.feature_significance import FeatureSignificance
from clustering_qa.datasets.feature_table import FeatureTable, SampleLabels

_MIN_GROUPS_FOR_KRUSKAL = 2
_REPORT_FDR_THRESHOLDS = (0.05, 0.01, 0.001)
_ETA_FORMAT = ".6g"
_NA_CELL = "n/a"


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


def clustering_agreement_report(
    partition_ids: list[str],
    p_adj_threshold: float,
    n_selected: int,
    *partition_scores: FeatureSignificance,
) -> str:
    """Render a markdown summary report of the clustering agreement run.

    The report contains the significant gene counts per partition (``p_adj``
    below each of 0.05, 0.01, 0.001), the total number of tested genes, the
    selection parameters, per-partition eta-squared summary statistics
    (min, max, median, mean, 25p, 75p, variance) across all tested
    features, and the hypothetical consensus sizes obtained by
    intersecting the significant gene sets at each threshold.

    Args:
        partition_ids: Partition names, order-matched to ``partition_scores``.
        p_adj_threshold: Selection FDR threshold used for the consensus.
        n_selected: Maximum number of features selected per partition.
        partition_scores: One :class:`FeatureSignificance` per partition.

    Returns:
        The report as a markdown string.
    """
    if len(partition_ids) != len(partition_scores):
        msg = (
            f"got {len(partition_ids)} partition ids for "
            f"{len(partition_scores)} score tables"
        )
        raise ValueError(msg)

    per_partition = [
        {
            threshold: set(
                scores.data.filter(pl.col("p_adj") < threshold)["feature"].to_list()
            )
            for threshold in _REPORT_FDR_THRESHOLDS
        }
        for scores in partition_scores
    ]
    consensus = {
        threshold: (
            set.intersection(*(sets[threshold] for sets in per_partition))
            if per_partition
            else set()
        )
        for threshold in _REPORT_FDR_THRESHOLDS
    }
    total_genes = partition_scores[0].data.height if partition_scores else 0
    threshold_labels = [f"FDR < {t}" for t in _REPORT_FDR_THRESHOLDS]

    lines = [
        "# Clustering agreement report",
        "",
        "## Run parameters",
        "",
        f"- Total genes tested per cohort: {total_genes}",
        f"- Max genes selected per cohort (`n_selected`): {n_selected}",
        f"- Selection FDR threshold (`p_adj_threshold`): {p_adj_threshold}",
        "",
        "## Significant genes per cohort",
        "",
        "| Cohort | " + " | ".join(threshold_labels) + " |",
        "| --- | " + " | ".join("---" for _ in _REPORT_FDR_THRESHOLDS) + " |",
    ]
    for pid, counts in zip(partition_ids, per_partition, strict=True):
        cells = " | ".join(str(len(counts[t])) for t in _REPORT_FDR_THRESHOLDS)
        lines.append(f"| {pid} | {cells} |")
    lines += [
        "",
        "## Eta-squared summary per cohort",
        "",
        "Descriptive statistics of eta-squared across all tested features per",
        "cohort, excluding features with non-finite eta-squared. Variance is the",
        "population variance (ddof=0).",
        "",
        "| Cohort | Min | Max | Median | Mean | 25p | 75p | Variance |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for pid, scores in zip(partition_ids, partition_scores, strict=True):
        cells = " | ".join(_eta_squared_summary_cells(scores))
        lines.append(f"| {pid} | {cells} |")
    lines += [
        "",
        "## Consensus genes by FDR threshold",
        "",
        "Intersection of the significant gene sets across cohorts. Hypothetical:",
        "the consensus output selects the top `n_selected` genes per cohort by",
        "eta-squared instead.",
        "",
        "| Threshold | Consensus genes |",
        "| --- | --- |",
    ]
    lines += [
        f"| {label} | {len(consensus[t])} |"
        for t, label in zip(_REPORT_FDR_THRESHOLDS, threshold_labels, strict=True)
    ]
    return "\n".join(lines) + "\n"


def _eta_squared_summary_cells(scores: FeatureSignificance) -> list[str]:
    """Format one cohort's eta-squared summary as report table cells.

    Returns the min, max, median, mean, 25th and 75th percentiles (linear
    interpolation), and population variance of the finite eta-squared
    values, each formatted with 6 significant digits; ``n/a`` cells when
    no feature has a finite eta-squared.
    """
    finite = scores.data.filter(pl.col("eta_squared").is_finite())
    if finite.is_empty():
        return [_NA_CELL] * 7
    stats = finite.select(
        pl.col("eta_squared").min().alias("min"),
        pl.col("eta_squared").max().alias("max"),
        pl.col("eta_squared").median().alias("median"),
        pl.col("eta_squared").mean().alias("mean"),
        pl.col("eta_squared").quantile(0.25, interpolation="linear").alias("p25"),
        pl.col("eta_squared").quantile(0.75, interpolation="linear").alias("p75"),
        pl.col("eta_squared").var(ddof=0).alias("variance"),
    ).row(0)
    return [f"{value:{_ETA_FORMAT}}" for value in stats]


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
