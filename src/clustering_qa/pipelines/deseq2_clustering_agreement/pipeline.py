"""DESeq2 clustering agreement pipeline.

Input: per-partition feature tables holding raw RNA-seq counts and
    per-partition labels/strata.
Method: run a DESeq2 likelihood ratio test on every feature (``~ label``
    vs reduced ``~ 1``, so the p-value asks whether the feature differs
    across any label group) via the R project in
    ``r/deseq2_clustering_agreement``; adjusted p-values and independent
    filtering come from ``DESeq2::results()``; the effect size is the
    pairwise log2 fold change between cluster labels, shrunk with ashr and
    reduced to a lower confidence bound.
Output: the ``feature_significance__{pid}`` tables and two consensus lists
    selected by ``p_adj`` and the fold-change gap — ``..._global`` (every
    cluster pair separated) and ``..._local`` (at least one pair separated).
"""

from kedro.pipeline import Pipeline, node

from clustering_qa.pipelines.clustering_agreement.nodes import (
    select_consensus_features,
)

from .nodes import deseq2_lrt_per_feature


def create_pipeline(partition_ids: list[str]) -> Pipeline:
    """Create the DESeq2 clustering agreement pipeline."""
    scores = [f"feature_significance__{pid}" for pid in partition_ids]
    pipeline_nodes = [
        node(
            func=deseq2_lrt_per_feature,
            inputs=[
                f"raw_clustering_features__{pid}",
                f"raw_clustering_labels__{pid}",
                "params:clustering_agreement.p_adj_threshold",
                "params:clustering_agreement.n_features",
                "params:deseq2_clustering_agreement.r_project",
                "params:deseq2_clustering_agreement.nproc",
            ],
            outputs=f"feature_significance__{pid}",
        )
        for pid in partition_ids
    ]
    pipeline_nodes.append(
        node(
            func=select_consensus_features,
            inputs=[
                "params:clustering_agreement.p_adj_threshold",
                "params:clustering_agreement.lfc_gap_threshold",
                *scores,
            ],
            outputs=[
                "clustering_agreement_features_global",
                "clustering_agreement_features_local",
            ],
        )
    )
    return Pipeline(pipeline_nodes)
