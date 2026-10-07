"""Structural tests for the clustering_agreement_summary pipeline."""

from kedro.pipeline import Node, Pipeline

from clustering_qa.pipelines.clustering_agreement_summary import create_pipeline


class TestCreatePipeline:
    """Tests for the ``create_pipeline`` factory."""

    @staticmethod
    def _nodes_by_name(pipeline: Pipeline) -> dict[str, Node]:
        return {node.name: node for node in pipeline.nodes}

    def test_one_node_per_output(self) -> None:
        pipeline = create_pipeline(["part1", "part2", "part3"])
        assert len(pipeline.nodes) == 2
        assert {node.name for node in pipeline.nodes} == {
            "clustering_agreement_report",
            "pairwise_jaccard_summary",
        }

    def test_report_node_reads_thresholds_and_scores(self) -> None:
        pipeline = create_pipeline(["part1", "part2"])
        report_node = self._nodes_by_name(pipeline)["clustering_agreement_report"]
        assert list(report_node.inputs) == [
            "params:partition_ids",
            "params:clustering_agreement.p_adj_threshold",
            "params:clustering_agreement.lfc_gap_threshold",
            "feature_significance__part1",
            "feature_significance__part2",
        ]
        assert report_node.outputs == ["clustering_agreement_report"]

    def test_jaccard_node_writes_report_and_plots(self) -> None:
        pipeline = create_pipeline(["part1", "part2"])
        jaccard_node = self._nodes_by_name(pipeline)["pairwise_jaccard_summary"]
        assert list(jaccard_node.inputs) == [
            "params:partition_ids",
            "params:clustering_agreement.lfc_gap_threshold",
            "feature_significance__part1",
            "feature_significance__part2",
        ]
        assert list(jaccard_node.outputs) == [
            "jaccard_summary_report",
            "jaccard_plot",
            "jaccard_global_plot",
            "jaccard_local_plot",
        ]

    def test_empty_partition_list_still_summarizes(self) -> None:
        pipeline = create_pipeline([])
        report_node = self._nodes_by_name(pipeline)["clustering_agreement_report"]
        assert list(report_node.inputs) == [
            "params:partition_ids",
            "params:clustering_agreement.p_adj_threshold",
            "params:clustering_agreement.lfc_gap_threshold",
        ]
