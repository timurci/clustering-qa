"""Dataset definitions for clustering."""

from typing import TypedDict


class ClusterLabelScore(TypedDict):
    """Represents the ARI and NMI scores for a clustering trial."""

    ari: float
    nmi: float
