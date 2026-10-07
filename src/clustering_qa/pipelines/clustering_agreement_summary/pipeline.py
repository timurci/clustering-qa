"""Clustering agreement summary pipeline.

Input: the per-partition ``feature_significance__{pid}`` tables produced by
    the ``deseq2_clustering_agreement`` pipeline.
Method: render the run report and summarize pairwise cohort overlap of the
    significant gene sets (``p_adj`` below 0.05 / 0.01 / 0.001) as Jaccard
    matrices and heatmaps — once ignoring effect sizes, once requiring the
    global gap ``min_gap`` at or above ``clustering_agreement.
    lfc_gap_threshold``, and once requiring the local gap ``max_gap``.
Output: the markdown report, the Jaccard summary report, and three heatmap
    figures under ``data/08_reporting/clustering_agreement/``.
"""

from kedro.pipeline import Pipeline, node

from clustering_qa.pipelines.clustering_agreement.nodes import (
    clustering_agreement_report,
    pairwise_jaccard_summary,
)


def create_pipeline(partition_ids: list[str]) -> Pipeline:
    """Create the clustering agreement summary pipeline."""
    scores = [f"feature_significance__{pid}" for pid in partition_ids]
    return Pipeline(
        [
            node(
                func=clustering_agreement_report,
                inputs=[
                    "params:partition_ids",
                    "params:clustering_agreement.p_adj_threshold",
                    "params:clustering_agreement.lfc_gap_threshold",
                    *scores,
                ],
                outputs="clustering_agreement_report",
                name="clustering_agreement_report",
            ),
            node(
                func=pairwise_jaccard_summary,
                inputs=[
                    "params:partition_ids",
                    "params:clustering_agreement.lfc_gap_threshold",
                    *scores,
                ],
                outputs=[
                    "jaccard_summary_report",
                    "jaccard_plot",
                    "jaccard_global_plot",
                    "jaccard_local_plot",
                ],
                name="pairwise_jaccard_summary",
            ),
        ]
    )
