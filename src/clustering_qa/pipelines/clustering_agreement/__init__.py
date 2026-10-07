"""Shared clustering agreement nodes.

``select_consensus_features`` serves the ``deseq2_clustering_agreement``
pipeline; ``clustering_agreement_report`` and ``pairwise_jaccard_summary``
serve the ``clustering_agreement_summary`` stage. The registry stacks the
method pipeline with that stage under the ``clustering_agreement`` name.
This package does not define a pipeline of its own.
"""

from .nodes import (
    clustering_agreement_report,
    pairwise_jaccard_summary,
    select_consensus_features,
)

__all__ = [
    "clustering_agreement_report",
    "pairwise_jaccard_summary",
    "select_consensus_features",
]
