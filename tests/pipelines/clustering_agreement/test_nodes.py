"""Unit tests for the shared clustering agreement nodes."""

import polars as pl
import pytest

from clustering_qa.datasets.feature_significance import FeatureSignificance
from clustering_qa.pipelines.clustering_agreement.nodes import (
    _EFFECT_DESCRIPTION,
    clustering_agreement_report,
    pairwise_jaccard_summary,
    select_consensus_features,
)

_GAP_THRESHOLD = 0.5
_SECTION_HEADINGS = (
    "Run parameters",
    "Significant genes per cohort",
    "Fold-change gap summary per cohort",
    "Consensus genes by FDR threshold",
)


def _scores(rows: dict[str, tuple[float, float, float]]) -> FeatureSignificance:
    """Builds scores from ``feature -> (p_adj, min_gap, max_gap)``."""
    return FeatureSignificance(
        data=pl.DataFrame(
            {
                "feature": list(rows),
                "p_value": [p for p, _, _ in rows.values()],
                "p_adj": [p for p, _, _ in rows.values()],
                "min_gap": [lo for _, lo, _ in rows.values()],
                "max_gap": [hi for _, _, hi in rows.values()],
            }
        )
    )


class TestSelectConsensusFeatures:
    """Tests for the ``select_consensus_features`` node function."""

    def test_global_uses_min_gap_and_intersects(self) -> None:
        a = _scores({"x": (0.01, 0.6, 1.0), "y": (0.01, 0.2, 0.9)})
        b = _scores({"x": (0.01, 0.6, 1.0), "y": (0.01, 0.2, 0.9)})
        global_features, _ = select_consensus_features(0.05, _GAP_THRESHOLD, a, b)
        assert global_features == ["x"]

    def test_local_uses_max_gap_and_intersects(self) -> None:
        a = _scores({"x": (0.01, 0.6, 1.0), "y": (0.01, 0.2, 0.9)})
        b = _scores({"x": (0.01, 0.6, 1.0), "y": (0.01, 0.2, 0.9)})
        _, local_features = select_consensus_features(0.05, _GAP_THRESHOLD, a, b)
        assert local_features == ["x", "y"]

    def test_fdr_gate_excludes_nonsignificant(self) -> None:
        a = _scores({"x": (0.01, 0.6, 1.0), "y": (0.2, 0.6, 1.0)})
        b = _scores({"x": (0.01, 0.6, 1.0), "y": (0.2, 0.6, 1.0)})
        global_features, local_features = select_consensus_features(
            0.05, _GAP_THRESHOLD, a, b
        )
        assert global_features == ["x"]
        assert local_features == ["x"]

    def test_non_finite_gaps_never_pass(self) -> None:
        a = _scores(
            {
                "x": (0.01, 0.6, 1.0),
                "y": (0.01, float("nan"), float("nan")),
                "z": (0.01, float("inf"), float("inf")),
            }
        )
        b = _scores({"x": (0.01, 0.6, 1.0)})
        global_features, local_features = select_consensus_features(
            0.05, _GAP_THRESHOLD, a, b
        )
        assert global_features == ["x"]
        assert local_features == ["x"]

    def test_no_partitions_returns_empty_pair(self) -> None:
        assert select_consensus_features(0.05, _GAP_THRESHOLD) == ([], [])

    def test_intersection_is_empty_when_disjoint(self) -> None:
        a = _scores({"x": (0.01, 0.6, 1.0)})
        b = _scores({"y": (0.01, 0.6, 1.0)})
        assert select_consensus_features(0.05, _GAP_THRESHOLD, a, b) == ([], [])


class TestClusteringAgreementReport:
    """Tests for the ``clustering_agreement_report`` node function."""

    def test_renders_test_description(self) -> None:
        a = _scores({"x": (0.0005, 0.6, 1.0)})
        report = clustering_agreement_report(["a"], 0.05, _GAP_THRESHOLD, a)
        assert f"- Test: {_EFFECT_DESCRIPTION}" in report

    def test_renders_document_title_and_sections(self) -> None:
        a = _scores({"x": (0.0005, 0.6, 1.0)})
        lines = clustering_agreement_report(["a"], 0.05, _GAP_THRESHOLD, a).splitlines()
        assert lines[0] == "# Clustering agreement report"
        for section in _SECTION_HEADINGS:
            assert f"## {section}" in lines

    def test_renders_counts_totals_and_consensus_sizes(self) -> None:
        a = _scores(
            {
                "x": (0.0005, 0.6, 1.0),
                "y": (0.005, 0.3, 0.8),
                "z": (0.04, 0.1, 0.2),
            }
        )
        b = _scores(
            {
                "x": (0.0005, 0.6, 1.0),
                "y": (0.02, 0.2, 0.5),
                "w": (0.3, 0.9, 0.9),
            }
        )
        report = clustering_agreement_report(["a", "b"], 0.05, _GAP_THRESHOLD, a, b)
        assert "| a | 3 | 2 | 1 |" in report
        assert "| b | 2 | 1 | 1 |" in report
        assert "- Total genes tested per cohort: 3" in report
        assert "- Selection FDR threshold (`p_adj_threshold`): 0.05" in report
        assert "- Minimum fold-change gap (`lfc_gap_threshold`): 0.5" in report
        # Global: {x} at every threshold. Local: {x, y} only at FDR < 0.05.
        assert "| FDR < 0.05 | 1 | 2 |" in report
        assert "| FDR < 0.01 | 1 | 1 |" in report
        assert "| FDR < 0.001 | 1 | 1 |" in report

    def test_renders_gap_summary_per_cohort(self) -> None:
        a = _scores(
            {
                "x": (0.0005, 0.6, 1.0),
                "y": (0.005, 0.3, 0.8),
                "z": (0.04, 0.1, 0.2),
            }
        )
        report = clustering_agreement_report(["a"], 0.05, _GAP_THRESHOLD, a)
        assert (
            "| Cohort | Gap | Min | Max | Median | Mean | 25p | 75p | Variance |"
            in report
        )
        assert "| a | min | 0.1 | 0.6 |" in report
        assert "| a | max | 0.2 | 1 |" in report

    def test_gap_summary_skips_nonfinite_values(self) -> None:
        a = _scores(
            {
                "x": (0.0005, 0.6, 1.0),
                "y": (0.005, float("nan"), float("nan")),
            }
        )
        report = clustering_agreement_report(["a"], 0.05, _GAP_THRESHOLD, a)
        assert "| a | min | 0.6 | 0.6 | 0.6 | 0.6 | 0.6 | 0.6 | 0 |" in report

    def test_gap_summary_all_nonfinite_is_na(self) -> None:
        a = _scores({"x": (0.0005, float("nan"), float("nan"))})
        report = clustering_agreement_report(["a"], 0.05, _GAP_THRESHOLD, a)
        assert "| a | min | n/a | n/a | n/a | n/a | n/a | n/a | n/a |" in report
        assert "| a | max | n/a | n/a | n/a | n/a | n/a | n/a | n/a |" in report

    def test_renders_empty_run(self) -> None:
        report = clustering_agreement_report([], 0.05, _GAP_THRESHOLD)
        assert "| Cohort | FDR < 0.05 | FDR < 0.01 | FDR < 0.001 |" in report
        assert (
            "| Cohort | Gap | Min | Max | Median | Mean | 25p | 75p | Variance |"
            in report
        )
        assert "| Threshold | Global consensus | Local consensus |" in report

    def test_partition_id_mismatch_raises(self) -> None:
        a = _scores({"x": (0.0005, 0.6, 1.0)})
        with pytest.raises(ValueError, match="partition ids"):
            clustering_agreement_report(["a", "b"], 0.05, _GAP_THRESHOLD, a)


class TestPairwiseJaccardSummary:
    """Tests for the ``pairwise_jaccard_summary`` node function."""

    @staticmethod
    def _three_cohorts() -> list[FeatureSignificance]:
        return [
            _scores(
                {
                    "x": (0.0005, 0.6, 1.0),
                    "y": (0.005, 0.3, 0.8),
                    "z": (0.04, 0.1, 0.2),
                }
            ),
            _scores(
                {
                    "x": (0.0005, 0.6, 1.0),
                    "y": (0.005, 0.3, 0.8),
                    "w": (0.04, 0.1, 0.2),
                }
            ),
            _scores({"x": (0.0005, 0.6, 1.0)}),
        ]

    def test_renders_jaccard_matrices_per_threshold(self) -> None:
        report, _, _, _ = pairwise_jaccard_summary(
            ["a", "b", "c"], _GAP_THRESHOLD, *self._three_cohorts()
        )
        assert "# Pairwise Jaccard summary" in report
        assert f"- Test: {_EFFECT_DESCRIPTION}" in report
        assert "effect sizes are" in report
        assert "ignored" in report
        # Set sizes: a={x,y,z}, b={x,y,w}, c={x} at 0.05; {x,y} at 0.01; {x} at 0.001.
        assert "| a | 3 | 2 | 1 |" in report
        assert "| b | 3 | 2 | 1 |" in report
        assert "| c | 1 | 1 | 1 |" in report
        # Jaccard at FDR < 0.05: |ab| = 2/4, |ac| = |bc| = 1/3.
        assert "| a | 1.000 | 0.500 | 0.333 |" in report
        assert "| b | 0.500 | 1.000 | 0.333 |" in report
        assert "| c | 0.333 | 0.333 | 1.000 |" in report

    def test_sets_ignore_effect_sizes(self) -> None:
        a = _scores({"x": (0.0005, 0.01, 0.99), "big": (0.5, 0.99, 0.99)})
        b = _scores({"x": (0.0005, 0.99, 0.99)})
        report, figure, _, _ = pairwise_jaccard_summary(
            ["a", "b"], _GAP_THRESHOLD, a, b
        )
        # `big` has the largest gap but is not significant, so both sets hold
        # `x` alone: Jaccard 1.000 with one shared gene, not 0.000 against `big`.
        assert "| a | 1 | 1 | 1 |" in report
        assert "| b | 1 | 1 | 1 |" in report
        assert [text.get_text() for text in figure.axes[0].texts] == ["1.000"] * 4
        assert [text.get_text() for text in figure.axes[3].texts] == ["1"] * 4

    def test_global_figure_applies_min_gap_filter(self) -> None:
        a = _scores({"x": (0.0005, 0.6, 1.0), "y": (0.0005, 0.2, 0.9)})
        b = _scores({"x": (0.0005, 0.6, 1.0), "y": (0.0005, 0.2, 0.9)})
        _, figure, global_figure, _ = pairwise_jaccard_summary(
            ["a", "b"], _GAP_THRESHOLD, a, b
        )
        # FDR alone: a == b == {x, y}.
        assert [text.get_text() for text in figure.axes[3].texts] == ["2"] * 4
        # min_gap >= 0.5 drops `y`: a == b == {x}.
        assert [text.get_text() for text in global_figure.axes[3].texts] == ["1"] * 4
        assert [ax.get_title() for ax in global_figure.axes[:3]] == [
            "FDR < 0.05, min_gap >= 0.5",
            "FDR < 0.01, min_gap >= 0.5",
            "FDR < 0.001, min_gap >= 0.5",
        ]

    def test_local_figure_applies_max_gap_filter(self) -> None:
        a = _scores({"x": (0.0005, 0.6, 1.0), "y": (0.0005, 0.2, 0.9)})
        b = _scores({"x": (0.0005, 0.6, 1.0), "y": (0.0005, 0.2, 0.9)})
        _, _, _, local_figure = pairwise_jaccard_summary(
            ["a", "b"], _GAP_THRESHOLD, a, b
        )
        # max_gap >= 0.5 keeps both: a == b == {x, y}.
        assert [text.get_text() for text in local_figure.axes[3].texts] == ["2"] * 4
        assert [ax.get_title() for ax in local_figure.axes[:3]] == [
            "FDR < 0.05, max_gap >= 0.5",
            "FDR < 0.01, max_gap >= 0.5",
            "FDR < 0.001, max_gap >= 0.5",
        ]

    def test_effect_figures_exclude_non_finite_gaps(self) -> None:
        """``NaN`` and ``inf`` compare true against ``>=`` in Polars."""
        a = _scores(
            {"x": (0.0005, 0.6, 1.0), "y": (0.0005, float("nan"), float("nan"))}
        )
        b = _scores(
            {"x": (0.0005, 0.6, 1.0), "y": (0.0005, float("inf"), float("inf"))}
        )
        _, _, global_figure, local_figure = pairwise_jaccard_summary(
            ["a", "b"], _GAP_THRESHOLD, a, b
        )
        assert [text.get_text() for text in global_figure.axes[3].texts] == ["1"] * 4
        assert [text.get_text() for text in local_figure.axes[3].texts] == ["1"] * 4

    def test_empty_sets_yield_na_cells(self) -> None:
        a = _scores({"x": (0.5, 0.6, 1.0)})
        b = _scores({"y": (0.5, 0.6, 1.0)})
        report, figure, global_figure, local_figure = pairwise_jaccard_summary(
            ["a", "b"], _GAP_THRESHOLD, a, b
        )
        assert "| a | 0 | 0 | 0 |" in report
        assert "| a | n/a | n/a |" in report
        assert "| b | n/a | n/a |" in report
        # Two empty sets intersect in nothing: n/a Jaccard, zero shared genes.
        for plot in (figure, global_figure, local_figure):
            assert [text.get_text() for text in plot.axes[0].texts] == ["n/a"] * 4
            assert [text.get_text() for text in plot.axes[3].texts] == ["0"] * 4

    def test_no_partitions_renders_headers_only(self) -> None:
        report, figure, global_figure, local_figure = pairwise_jaccard_summary(
            [], _GAP_THRESHOLD
        )
        assert "| Cohort | FDR < 0.05 | FDR < 0.01 | FDR < 0.001 |" in report
        assert len(figure.axes) == 6
        assert len(global_figure.axes) == 6
        assert len(local_figure.axes) == 6

    def test_figure_rows_annotate_jaccard_and_intersection_counts(self) -> None:
        _, figure, _, _ = pairwise_jaccard_summary(
            ["a", "b"], _GAP_THRESHOLD, *self._three_cohorts()[:2]
        )
        assert [ax.get_title() for ax in figure.axes[:3]] == [
            "FDR < 0.05",
            "FDR < 0.01",
            "FDR < 0.001",
        ]
        assert [ax.get_title() for ax in figure.axes[3:]] == ["", "", ""]
        assert figure.axes[0].get_ylabel() == "Jaccard index"
        assert figure.axes[3].get_ylabel() == "shared genes"
        # a = {x, y, z} and b = {x, y, w} at 0.05 -> 2/4 with 2 shared genes.
        assert [text.get_text() for text in figure.axes[0].texts] == [
            "1.000",
            "0.500",
            "0.500",
            "1.000",
        ]
        assert [text.get_text() for text in figure.axes[3].texts] == [
            "3",
            "2",
            "2",
            "3",
        ]

    def test_annotation_colors_contrast_with_cell_colors(self) -> None:
        """Regression: viridis is bright at high values."""
        a, _, c = self._three_cohorts()
        _, figure, _, _ = pairwise_jaccard_summary(["a", "c"], _GAP_THRESHOLD, a, c)
        expected = ["black", "white", "white", "black"]
        assert [text.get_color() for text in figure.axes[0].texts] == expected

    def test_partition_id_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="partition ids"):
            pairwise_jaccard_summary(
                ["a", "b"], _GAP_THRESHOLD, self._three_cohorts()[0]
            )
