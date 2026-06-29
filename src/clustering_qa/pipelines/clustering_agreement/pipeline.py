"""Clustering agreement pipeline.

Input: per-partition feature embeddings and per-partition labels/strata.
Method: run a Kruskal-Wallis H-test on every feature (against the labels
    as the grouping variable), Benjamini-Hochberg-adjust the p-values
    within each partition, then take the intersection of significant
    features (p_adj < threshold) across partitions.
Output: per-partition feature significance tables and a list of feature
    names significant in every partition.
"""

from kedro.pipeline import Pipeline, node

from .nodes import intersect_significant_features, kruskal_wallis_per_feature


def create_pipeline(partition_ids: list[str]) -> Pipeline:
    """Create the clustering agreement pipeline."""
    pipeline_nodes = [
        node(
            func=kruskal_wallis_per_feature,
            inputs=[
                f"raw_clustering_features__{pid}",
                f"raw_clustering_labels__{pid}",
            ],
            outputs=f"feature_significance__{pid}",
        )
        for pid in partition_ids
    ]
    pipeline_nodes.append(
        node(
            func=intersect_significant_features,
            inputs=[
                "params:clustering_agreement.p_adj_threshold",
                *[f"feature_significance__{pid}" for pid in partition_ids],
            ],
            outputs="clustering_agreement_features",
        )
    )
    return Pipeline(pipeline_nodes)
