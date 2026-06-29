"""Project pipelines."""

from kedro.pipeline import Pipeline

from clustering_qa import pipelines


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    clustering_stability = pipelines.clustering_stability.create_pipeline()
    pipeline_dict: dict[str, Pipeline] = {
        "clustering_stability": clustering_stability,
        "__default__": clustering_stability,
    }
    return pipeline_dict
