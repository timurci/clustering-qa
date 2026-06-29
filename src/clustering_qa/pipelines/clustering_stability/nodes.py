"""Nodes for the clustering stability pipeline."""

from itertools import combinations

import polars as pl
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from clustering_qa.datasets.clustering import ClusterLabelScore
from clustering_qa.datasets.embedding import EmbeddingLabels, Embeddings


def kmeans_clustering(
    embeddings: Embeddings, n_clusters: int, seed: int | None = None
) -> EmbeddingLabels:
    """Performs K-means clustering on the given embeddings."""
    model = KMeans(n_clusters=n_clusters, random_state=seed).fit(embeddings.data)
    labels: list[str] = [str(label) for label in model.labels_.tolist()]
    labels_df = pl.DataFrame(
        {"sample_id": embeddings.data["sample_id"], "label": labels},
        schema={
            "sample_id": embeddings.data.schema["sample_id"],
            "label": pl.Categorical,
        },
    )
    return EmbeddingLabels(data=labels_df)


def assess_stability(*trials: EmbeddingLabels) -> ClusterLabelScore:
    """Assesses the stability of clustering trials by comparing their labels.

    Returns:
        Pairwise mean ARI and NMI scores across multiple clustering trials.
    """

    def score(trial1: EmbeddingLabels, trial2: EmbeddingLabels) -> ClusterLabelScore:
        ari = adjusted_rand_score(trial1.data["label"], trial2.data["label"])
        nmi = normalized_mutual_info_score(trial1.data["label"], trial2.data["label"])
        return {"ari": ari, "nmi": nmi}

    pairwise_scores: list[ClusterLabelScore] = [
        score(trial1, trial2) for trial1, trial2 in combinations(trials, 2)
    ]

    mean: ClusterLabelScore = {
        "ari": sum(score["ari"] for score in pairwise_scores) / len(pairwise_scores),
        "nmi": sum(score["nmi"] for score in pairwise_scores) / len(pairwise_scores),
    }

    return mean
