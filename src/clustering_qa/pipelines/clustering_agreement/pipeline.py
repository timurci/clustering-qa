"""Clustering agreement pipeline.

Input: per-partition feature tables and per-partition labels/strata.
Method: run a Kruskal-Wallis H-test on every feature (against the labels
    as the grouping variable) for p-values and eta-squared effect sizes,
    Benjamini-Hochberg-adjust the p-values within each partition, select
    the top ``n_selected`` features per partition by eta-squared among the
    FDR survivors (p_adj < threshold), and take the intersection of the
    per-partition selections as the consensus feature set.
Output: per-partition feature significance tables, the consensus list of
    feature names prioritized in every partition, and a markdown summary
    report of significant gene counts.
"""

from kedro.pipeline import Pipeline, node

from .nodes import (
    clustering_agreement_report,
    kruskal_wallis_per_feature,
    select_consensus_features,
)


def create_pipeline(partition_ids: list[str]) -> Pipeline:
    """Create the clustering agreement pipeline."""
    pipeline_nodes = [
        node(
            func=kruskal_wallis_per_feature,
            inputs=[
                f"raw_clustering_features__{pid}",
                f"raw_clustering_labels__{pid}",
                "params:clustering_agreement.n_features",
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
                "params:clustering_agreement.n_selected",
                *[f"feature_significance__{pid}" for pid in partition_ids],
            ],
            outputs="clustering_agreement_features",
        )
    )
    pipeline_nodes.append(
        node(
            func=clustering_agreement_report,
            inputs=[
                "params:partition_ids",
                "params:clustering_agreement.p_adj_threshold",
                "params:clustering_agreement.n_selected",
                *[f"feature_significance__{pid}" for pid in partition_ids],
            ],
            outputs="clustering_agreement_report",
        )
    )
    return Pipeline(pipeline_nodes)
