"""Clustering stability pipeline.

Input: takes a partitioned (i.e multiple) embedded input(s).
Method: apply k-means clustering with multiple trials and then compute the agreement
    between the assignments and against pre-determined labels.
Output: mean pairwise label agreement metrics (ARI/NMI)
"""

from functools import partial
from typing import Any

from kedro.pipeline import Pipeline, node

from .nodes import assess_stability, kmeans_clustering


def create_pipeline(**kwargs: dict[str, Any]) -> Pipeline:
    """Create the clustering stability pipeline."""
    params: dict[str, Any] = kwargs["parameters"]
    partition_ids: list[str] = params["partition_ids"]
    n_trials: int = params["clustering_stability"]["n_trials"]

    pipeline_nodes = []

    for pid in partition_ids:
        # Stage 1: Generate clustering trials for each partition
        for trial in range(n_trials):
            pipeline_nodes.extend(
                [
                    node(
                        func=partial(kmeans_clustering, seed=trial),
                        inputs=[
                            f"raw_clustering_embeddings.{pid}",
                            "params:clustering_stability.n_clusters",
                        ],
                        outputs=f"clustering_labels.{pid}.trial_{trial}",
                    )
                ]
            )

        # Stage 2: Compute clustering stability metrics
        pipeline_nodes.append(
            node(
                func=assess_stability,
                inputs=f"clustering_labels.{pid}",
                outputs=f"clustering_stability_metrics.{pid}",
            )
        )

    return Pipeline(pipeline_nodes)
