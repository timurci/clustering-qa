"""Project pipelines."""

from pathlib import Path

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings
from kedro.pipeline import Pipeline

from clustering_qa import pipelines


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    conf_path = str(Path.cwd() / settings.CONF_SOURCE)
    conf_loader = OmegaConfigLoader(conf_source=conf_path)
    parameters = conf_loader["parameters"]

    partition_ids = parameters["partition_ids"]

    clustering_stability = pipelines.clustering_stability.create_pipeline(
        partition_ids=partition_ids,
        n_trials=parameters["clustering_stability"]["n_trials"],
    )
    clustering_agreement = pipelines.deseq2_clustering_agreement.create_pipeline(
        partition_ids=partition_ids,
    ) + pipelines.clustering_agreement_summary.create_pipeline(
        partition_ids=partition_ids,
    )
    pipeline_dict: dict[str, Pipeline] = {
        "clustering_stability": clustering_stability,
        "clustering_agreement": clustering_agreement,
        "__default__": clustering_stability + clustering_agreement,
    }
    return pipeline_dict
