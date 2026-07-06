"""Nodes for the clustering stability pipeline."""

from itertools import combinations

import polars as pl
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from clustering_qa.datasets.clustering import ClusterLabelScore
from clustering_qa.datasets.feature_table import FeatureTable, SampleLabels


def kmeans_clustering(
    features: FeatureTable, n_clusters: int, seed: int | None = None
) -> SampleLabels:
    """Performs K-means clustering on the given feature table."""
    model = KMeans(n_clusters=n_clusters, random_state=seed).fit(
        features.data.drop("sample_id")
    )
    labels: list[str] = [str(label) for label in model.labels_.tolist()]
    labels_df = pl.DataFrame(
        {
            "sample_id": features.data["sample_id"],
            "label": labels,
        },
        schema={
            "sample_id": features.data.schema["sample_id"],
            "label": pl.Categorical,
        },
    )
    return SampleLabels(data=labels_df)


def assess_stability(*trials: SampleLabels) -> ClusterLabelScore:
    """Assesses the stability of clustering trials by comparing their labels.

    Returns:
        Pairwise mean ARI and NMI scores across multiple clustering trials.
    """

    def score(trial1: SampleLabels, trial2: SampleLabels) -> ClusterLabelScore:
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
