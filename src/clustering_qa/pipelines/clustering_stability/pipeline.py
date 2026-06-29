"""Clustering stability pipeline.

Input: takes a partitioned (i.e multiple) embedded input(s).
Method: apply k-means clustering with multiple trials and then compute the agreement
    between the assignments and against pre-determined labels.
Output: mean pairwise label agreement metrics (ARI/NMI)
"""

from functools import partial

from kedro.pipeline import Pipeline, node

from .nodes import assess_stability, kmeans_clustering


def create_pipeline(partition_ids: list[str], n_trials: int) -> Pipeline:
    """Create the clustering stability pipeline."""
    pipeline_nodes = []

    for pid in partition_ids:
        # Stage 1: Generate clustering trials for each partition
        pipeline_nodes.extend(
            [
                node(
                    func=partial(kmeans_clustering, seed=trial),
                    inputs=[
                        f"raw_clustering_embeddings__{pid}",
                        "params:clustering_stability.n_clusters",
                    ],
                    outputs=f"intermediate_clustering_labels__{pid}__trial_{trial}",
                )
                for trial in range(n_trials)
            ]
        )

        # Stage 2: Compute clustering stability metrics. Take the list of
        # per-trial outputs as separate inputs so each MemoryDataset produced
        # by Stage 1 is consumed individually.
        pipeline_nodes.append(
            node(
                func=assess_stability,
                inputs=[
                    f"intermediate_clustering_labels__{pid}__trial_{t}"
                    for t in range(n_trials)
                ],
                outputs=f"clustering_stability_metrics__{pid}",
            )
        )

    return Pipeline(pipeline_nodes)
