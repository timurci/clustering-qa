"""Smoke tests for the Kedro entry point and the registered pipelines."""

from kedro.pipeline import Pipeline

from clustering_qa import pipelines as pipeline_packages
from clustering_qa.pipeline_registry import register_pipelines

_PARTITION_IDS = ["part1", "part2"]


def _outputs(pipeline: Pipeline) -> set[str]:
    """Return the dataset names a pipeline writes."""
    return {name for node in pipeline.nodes for name in node.outputs}


class TestKedroRun:
    """Smoke tests that exercise the Kedro run entry point."""

    def test_kedro_run_default_pipeline_is_not_empty(self):
        pipelines = register_pipelines()
        assert len(pipelines["__default__"].nodes) > 0

    def test_agreement_pipeline_writes_expected_datasets(self) -> None:
        deseq2 = pipeline_packages.deseq2_clustering_agreement.create_pipeline(
            _PARTITION_IDS
        )
        assert _outputs(deseq2) == {
            "feature_significance__part1",
            "feature_significance__part2",
            "clustering_agreement_features_global",
            "clustering_agreement_features_local",
        }

    def test_registered_agreement_pipeline_stacks_method_and_summary(self) -> None:
        outputs = _outputs(register_pipelines()["clustering_agreement"])
        assert {
            "clustering_agreement_features_global",
            "clustering_agreement_features_local",
            "clustering_agreement_report",
            "jaccard_summary_report",
            "jaccard_plot",
            "jaccard_global_plot",
            "jaccard_local_plot",
        } <= outputs
        assert any(name.startswith("feature_significance__") for name in outputs)
