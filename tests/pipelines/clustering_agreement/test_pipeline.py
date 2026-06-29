"""Structural tests for the clustering_agreement pipeline."""

from kedro.pipeline import Node, Pipeline

from clustering_qa.pipelines.clustering_agreement import create_pipeline


class TestCreatePipeline:
    """Tests for the ``create_pipeline`` factory."""

    @staticmethod
    def _nodes_by_output(pipeline: Pipeline) -> dict[str, Node]:
        return {n.outputs[0]: n for n in pipeline.nodes if len(n.outputs) == 1}

    def test_node_count_matches_partitions(self) -> None:
        partition_ids = ["part1", "part2", "part3"]
        pipeline = create_pipeline(partition_ids)
        # One KW node per partition plus a single intersection node.
        assert len(pipeline.nodes) == len(partition_ids) + 1

    def test_node_names_and_io(self) -> None:
        partition_ids = ["part1", "part2"]
        pipeline = create_pipeline(partition_ids)
        by_output = self._nodes_by_output(pipeline)

        for pid in partition_ids:
            node = by_output[f"feature_significance__{pid}"]
            assert list(node.inputs) == [
                f"raw_clustering_features__{pid}",
                f"raw_clustering_labels__{pid}",
            ]

        intersection_node = by_output["clustering_agreement_features"]
        assert list(intersection_node.inputs) == [
            "params:clustering_agreement.p_adj_threshold",
            "feature_significance__part1",
            "feature_significance__part2",
        ]

    def test_empty_partition_list_produces_only_intersection_node(self) -> None:
        pipeline = create_pipeline([])
        assert len(pipeline.nodes) == 1
        assert pipeline.nodes[0].outputs == ["clustering_agreement_features"]
        assert list(pipeline.nodes[0].inputs) == [
            "params:clustering_agreement.p_adj_threshold",
        ]
