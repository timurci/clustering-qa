"""Shared nodes for the clustering agreement pipeline.

Consensus selection, reporting, and overlap summaries operate on the
per-partition :class:`FeatureSignificance` tables produced by the
``deseq2_clustering_agreement`` pipeline. Each feature carries two effect
sizes: ``min_gap``, the smallest pairwise fold-change lower bound over the
cluster pairs (large only when every pair is separated), and ``max_gap``,
the largest such bound (large when at least one pair is separated).
"""

import math
from collections.abc import Callable, Iterable
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from clustering_qa.datasets.feature_significance import FeatureSignificance

_REPORT_FDR_THRESHOLDS = (0.05, 0.01, 0.001)
_GAP_FORMAT = ".6g"
_JACCARD_FORMAT = ".3f"
# Jaccard value at which white and black annotations reach equal WCAG
# contrast on viridis (relative luminance 0.179). Below it the dark end of
# the colormap needs white text, above it the bright end needs black text.
_ANNOTATION_COLOR_CUTOFF = 0.435
_NA_CELL = "n/a"
_JACCARD_CMAP = "viridis"
_JACCARD_BAD_COLOR = "#e0e0e0"

# The per-feature test and the effect size the pipeline reports.
_EFFECT_DESCRIPTION = (
    "DESeq2 likelihood ratio test per feature (`~ label` vs `~ 1`) on raw "
    "counts; effect size = pairwise log2 fold change between cluster labels, "
    "shrunk with ashr and reduced to a lower confidence bound "
    "(`|lfc| - z * posterior SD`)"
)


def select_consensus_features(
    p_adj_threshold: float,
    lfc_gap_threshold: float,
    *partition_scores: FeatureSignificance,
) -> tuple[list[str], list[str]]:
    """Select per-partition features by FDR and fold-change gap.

    A feature is selected in a partition when ``p_adj < p_adj_threshold``
    and its gap reaches ``lfc_gap_threshold``. The global selection uses
    ``min_gap`` (every cluster pair separated), the local selection uses
    ``max_gap`` (at least one pair separated). Each consensus is the sorted
    intersection of the per-partition selections across partitions.

    Args:
        p_adj_threshold: Maximum adjusted p-value that still counts as
            significant.
        lfc_gap_threshold: Minimum pairwise fold-change lower bound.
        partition_scores: One :class:`FeatureSignificance` per partition.

    Returns:
        The global consensus and the local consensus, each sorted.
    """
    if not partition_scores:
        return [], []
    global_sets = [
        _selected_features(scores, p_adj_threshold, "min_gap", lfc_gap_threshold)
        for scores in partition_scores
    ]
    local_sets = [
        _selected_features(scores, p_adj_threshold, "max_gap", lfc_gap_threshold)
        for scores in partition_scores
    ]
    return (
        sorted(reduce(set.intersection, global_sets)),
        sorted(reduce(set.intersection, local_sets)),
    )


def _selected_features(
    scores: FeatureSignificance,
    p_adj_threshold: float,
    gap_column: str,
    lfc_gap_threshold: float,
) -> set[str]:
    """Return the features of one partition passing the FDR and gap filters."""
    survivors = scores.data.filter(
        (pl.col("p_adj") < p_adj_threshold)
        & pl.col(gap_column).is_finite()
        & (pl.col(gap_column) >= lfc_gap_threshold)
    )
    return set(survivors["feature"].to_list())


def clustering_agreement_report(
    partition_ids: list[str],
    p_adj_threshold: float,
    lfc_gap_threshold: float,
    *partition_scores: FeatureSignificance,
) -> str:
    """Render the markdown report of one clustering agreement run.

    The report names the per-feature test and its effect size, lists the
    significant gene counts per partition (``p_adj`` below each of 0.05, 0.01,
    0.001), the total number of tested genes, per-partition min-gap and
    max-gap summary statistics (min, max, median, mean, 25p, 75p, variance),
    and the global and local consensus sizes obtained by intersecting the
    FDR- and gap-filtered gene sets across partitions.

    Args:
        partition_ids: Partition names, order-matched to ``partition_scores``.
        p_adj_threshold: Selection FDR threshold.
        lfc_gap_threshold: Minimum pairwise fold-change lower bound.
        partition_scores: One :class:`FeatureSignificance` per partition.

    Returns:
        The report as a markdown string.

    Raises:
        ValueError: If the number of partitions and score tables differs.
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
    global_consensus = {
        threshold: _intersect(
            _selected_features(scores, threshold, "min_gap", lfc_gap_threshold)
            for scores in partition_scores
        )
        for threshold in _REPORT_FDR_THRESHOLDS
    }
    local_consensus = {
        threshold: _intersect(
            _selected_features(scores, threshold, "max_gap", lfc_gap_threshold)
            for scores in partition_scores
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
        f"- Test: {_EFFECT_DESCRIPTION}",
        f"- Total genes tested per cohort: {total_genes}",
        f"- Selection FDR threshold (`p_adj_threshold`): {p_adj_threshold}",
        f"- Minimum fold-change gap (`lfc_gap_threshold`): {lfc_gap_threshold}",
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
        "## Fold-change gap summary per cohort",
        "",
        "Descriptive statistics of the pairwise fold-change lower bounds across",
        "all tested features per cohort, excluding non-finite values. `min_gap`",
        "is the smallest bound over the cluster pairs, `max_gap` the largest.",
        "Variance is the population variance (ddof=0).",
        "",
        "| Cohort | Gap | Min | Max | Median | Mean | 25p | 75p | Variance |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for pid, scores in zip(partition_ids, partition_scores, strict=True):
        for gap_column, label in (("min_gap", "min"), ("max_gap", "max")):
            cells = " | ".join(_gap_summary_cells(scores, gap_column))
            lines.append(f"| {pid} | {label} | {cells} |")
    lines += [
        "",
        "## Consensus genes by FDR threshold",
        "",
        "Intersection of the significant gene sets across cohorts. `global`",
        "requires `min_gap` (the smallest pairwise gap) to reach the threshold",
        "in every cohort, `local` only `max_gap` (the largest).",
        "",
        "| Threshold | Global consensus | Local consensus |",
        "| --- | --- | --- |",
    ]
    lines += [
        f"| {label} | {len(global_consensus[t])} | {len(local_consensus[t])} |"
        for t, label in zip(_REPORT_FDR_THRESHOLDS, threshold_labels, strict=True)
    ]
    return "\n".join(lines) + "\n"


def pairwise_jaccard_summary(
    partition_ids: list[str],
    lfc_gap_threshold: float,
    *partition_scores: FeatureSignificance,
) -> tuple[str, Figure, Figure, Figure]:
    """Summarize pairwise cohort overlap of significant gene sets.

    For each FDR threshold the significant gene set of a partition is the set
    of features with ``p_adj < threshold``. The Jaccard index of two sets is
    the size of their intersection over the size of their union, and is
    ``NaN`` when both sets are empty. Each figure holds one column per FDR
    threshold, with the Jaccard index of every cohort pair on top and the raw
    intersection size of the same pair below. The second figure repeats the
    heatmaps with ``min_gap >= lfc_gap_threshold`` (global) required on top
    of the FDR threshold, the third with ``max_gap >= lfc_gap_threshold``
    (local).

    Args:
        partition_ids: Partition names, order-matched to ``partition_scores``.
        lfc_gap_threshold: Minimum pairwise fold-change lower bound of the
            effect-filtered figures.
        partition_scores: One :class:`FeatureSignificance` per partition.

    Returns:
        The summary report as a markdown string, the FDR-only figure, the
        global (min-gap) figure, and the local (max-gap) figure.

    Raises:
        ValueError: If the number of partitions and score tables differs.
    """
    if len(partition_ids) != len(partition_scores):
        msg = (
            f"got {len(partition_ids)} partition ids for "
            f"{len(partition_scores)} score tables"
        )
        raise ValueError(msg)
    sets_per_threshold = {
        threshold: _significant_sets(partition_scores, threshold)
        for threshold in _REPORT_FDR_THRESHOLDS
    }
    global_sets = {
        threshold: _significant_sets(
            partition_scores, threshold, "min_gap", lfc_gap_threshold
        )
        for threshold in _REPORT_FDR_THRESHOLDS
    }
    local_sets = {
        threshold: _significant_sets(
            partition_scores, threshold, "max_gap", lfc_gap_threshold
        )
        for threshold in _REPORT_FDR_THRESHOLDS
    }
    report = _jaccard_report(partition_ids, sets_per_threshold, lfc_gap_threshold)
    figure = _jaccard_figure(partition_ids, sets_per_threshold)
    global_figure = _jaccard_figure(
        partition_ids, global_sets, filter_label=f"min_gap >= {lfc_gap_threshold}"
    )
    local_figure = _jaccard_figure(
        partition_ids, local_sets, filter_label=f"max_gap >= {lfc_gap_threshold}"
    )
    return report, figure, global_figure, local_figure


def _intersect(sets: Iterable[set[str]]) -> set[str]:
    """Return the intersection of an iterable of feature sets."""
    materialized = list(sets)
    return set.intersection(*materialized) if materialized else set()


def _significant_sets(
    partition_scores: tuple[FeatureSignificance, ...],
    p_adj_threshold: float,
    gap_column: str | None = None,
    lfc_gap_threshold: float | None = None,
) -> list[set[str]]:
    """Return the significant feature set of every partition.

    A feature is significant when ``p_adj`` is below ``p_adj_threshold``
    and, if a gap column is given, its finite value reaches
    ``lfc_gap_threshold``.
    """
    sets = []
    for scores in partition_scores:
        significant = scores.data.filter(pl.col("p_adj") < p_adj_threshold)
        if gap_column is not None and lfc_gap_threshold is not None:
            significant = significant.filter(
                pl.col(gap_column).is_finite()
                & (pl.col(gap_column) >= lfc_gap_threshold)
            )
        sets.append(set(significant["feature"].to_list()))
    return sets


def _jaccard_cell(left: set[str], right: set[str]) -> str:
    """Format one pairwise Jaccard index; ``n/a`` for two empty sets."""
    _, value = _overlap(left, right)
    if math.isnan(value):
        return _NA_CELL
    return f"{value:{_JACCARD_FORMAT}}"


def _jaccard_report(
    partition_ids: list[str],
    sets_per_threshold: dict[float, list[set[str]]],
    lfc_gap_threshold: float,
) -> str:
    """Render the Jaccard summary report markdown."""
    threshold_labels = [f"FDR < {t}" for t in _REPORT_FDR_THRESHOLDS]
    lines = [
        "# Pairwise Jaccard summary",
        "",
        "## Run parameters",
        "",
        f"- Test: {_EFFECT_DESCRIPTION}",
        f"- Minimum fold-change gap of the effect-filtered figures: "
        f"{lfc_gap_threshold}",
        "- Jaccard index: size of the intersection over the size of the union of",
        "  significant gene sets (`p_adj` < threshold only; effect sizes are",
        "  ignored).",
        f"- {_NA_CELL}: both gene sets are empty at that threshold.",
        "",
        "## Significant genes per cohort",
        "",
        "| Cohort | " + " | ".join(threshold_labels) + " |",
        "| --- | " + " | ".join("---" for _ in _REPORT_FDR_THRESHOLDS) + " |",
    ]
    for index, pid in enumerate(partition_ids):
        counts = [len(sets_per_threshold[t][index]) for t in _REPORT_FDR_THRESHOLDS]
        lines.append(f"| {pid} | " + " | ".join(map(str, counts)) + " |")
    for threshold, label in zip(_REPORT_FDR_THRESHOLDS, threshold_labels, strict=True):
        sets = sets_per_threshold[threshold]
        lines += [
            "",
            f"## Pairwise Jaccard — {label}",
            "",
            "| Cohort | " + " | ".join(partition_ids) + " |",
            "| --- | " + " | ".join("---" for _ in partition_ids) + " |",
        ]
        for index, pid in enumerate(partition_ids):
            cells = [_jaccard_cell(sets[index], other) for other in sets]
            lines.append(f"| {pid} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _jaccard_figure(
    partition_ids: list[str],
    sets_per_threshold: dict[float, list[set[str]]],
    filter_label: str | None = None,
) -> Figure:
    """Render the Jaccard and raw intersection heatmaps of every threshold.

    The top row holds the Jaccard index of every cohort pair and the bottom
    row the raw intersection size of the same pair, each annotated with its
    value. The intersection row shares one scale across thresholds so that
    the panels stay comparable. ``filter_label`` only names the gap filter
    the sets already reflect in the panel titles.
    """
    n_thresholds = len(_REPORT_FDR_THRESHOLDS)
    figure, axes = plt.subplots(
        2, n_thresholds, figsize=(5 * n_thresholds, 10), squeeze=False
    )
    overlaps = {
        threshold: _overlap_matrices(sets_per_threshold[threshold])
        for threshold in _REPORT_FDR_THRESHOLDS
    }
    intersection_max = 1.0
    if partition_ids:
        intersection_max = max(
            1.0,
            *(float(intersections.max()) for _, intersections in overlaps.values()),
        )
    for column, threshold in enumerate(_REPORT_FDR_THRESHOLDS):
        jaccard, intersections = overlaps[threshold]
        title = f"FDR < {threshold}"
        if filter_label is not None:
            title += f", {filter_label}"
        axes[0, column].set_title(title)
        _heatmap_panel(axes[0, column], jaccard, partition_ids, 1.0, _jaccard_text)
        _heatmap_panel(
            axes[1, column], intersections, partition_ids, intersection_max, _count_text
        )
    axes[0, 0].set_ylabel("Jaccard index")
    axes[1, 0].set_ylabel("shared genes")
    figure.tight_layout()
    return figure


def _overlap_matrices(sets: list[set[str]]) -> tuple[np.ndarray, np.ndarray]:
    """Return the Jaccard and raw intersection matrices of one threshold.

    Both matrices are indexed by cohort pair; the Jaccard matrix holds
    ``NaN`` where both sets are empty, the intersection matrix the raw
    counts of the same pairs.
    """
    n = len(sets)
    jaccard = np.empty((n, n))
    intersections = np.empty((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            intersections[i, j], jaccard[i, j] = _overlap(sets[i], sets[j])
    return jaccard, intersections


def _heatmap_panel(
    ax: Axes,
    matrix: np.ndarray,
    partition_ids: list[str],
    vmax: float,
    annotate: Callable[[float], str],
) -> None:
    """Draw one annotated cohort-pair heatmap panel."""
    n = len(partition_ids)
    if not n:
        return
    cmap = plt.get_cmap(_JACCARD_CMAP).with_extremes(bad=_JACCARD_BAD_COLOR)
    ax.imshow(matrix, vmin=0.0, vmax=vmax, cmap=cmap, aspect="equal")
    for i in range(n):
        for j in range(n):
            value = matrix[i, j]
            normalized = value / vmax
            ax.text(
                j,
                i,
                annotate(value),
                ha="center",
                va="center",
                color=(
                    "black"
                    if math.isnan(normalized) or normalized > _ANNOTATION_COLOR_CUTOFF
                    else "white"
                ),
            )
    ax.set_xticks(range(n), partition_ids, rotation=45, ha="right")
    ax.set_yticks(range(n), partition_ids)


def _overlap(left: set[str], right: set[str]) -> tuple[int, float]:
    """Return the intersection size and Jaccard index of two sets.

    The Jaccard index is the intersection size over the union size, and is
    ``NaN`` when both sets are empty.
    """
    intersection = len(left & right)
    union = len(left | right)
    return intersection, (intersection / union if union else float("nan"))


def _jaccard_text(value: float) -> str:
    """Format one Jaccard annotation; ``n/a`` for two empty sets."""
    return _NA_CELL if math.isnan(value) else f"{value:{_JACCARD_FORMAT}}"


def _count_text(value: float) -> str:
    """Format one raw intersection count annotation."""
    return f"{int(value)}"


def _gap_summary_cells(scores: FeatureSignificance, gap_column: str) -> list[str]:
    """Format one cohort's fold-change gap summary as report table cells.

    Returns the min, max, median, mean, 25th and 75th percentiles (linear
    interpolation), and population variance of the finite gap values, each
    formatted with 6 significant digits; ``n/a`` cells when no feature has a
    finite value.
    """
    finite = scores.data.filter(pl.col(gap_column).is_finite())
    if finite.is_empty():
        return [_NA_CELL] * 7
    stats = finite.select(
        pl.col(gap_column).min().alias("min"),
        pl.col(gap_column).max().alias("max"),
        pl.col(gap_column).median().alias("median"),
        pl.col(gap_column).mean().alias("mean"),
        pl.col(gap_column).quantile(0.25, interpolation="linear").alias("p25"),
        pl.col(gap_column).quantile(0.75, interpolation="linear").alias("p75"),
        pl.col(gap_column).var(ddof=0).alias("variance"),
    ).row(0)
    return [f"{value:{_GAP_FORMAT}}" for value in stats]
