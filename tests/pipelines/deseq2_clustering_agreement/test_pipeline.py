"""Structural tests for the deseq2_clustering_agreement pipeline."""

from kedro.pipeline import Node, Pipeline

from clustering_qa.pipelines.deseq2_clustering_agreement import create_pipeline


class TestCreatePipeline:
    """Tests for the ``create_pipeline`` factory."""

    @staticmethod
    def _nodes_by_output(pipeline: Pipeline) -> dict[str, Node]:
        return {n.outputs[0]: n for n in pipeline.nodes if len(n.outputs) == 1}

    def test_node_count_matches_partitions(self) -> None:
        partition_ids = ["part1", "part2", "part3"]
        pipeline = create_pipeline(partition_ids)
        # One DESeq2 node per partition plus a consensus node.
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
                "params:clustering_agreement.p_adj_threshold",
                "params:clustering_agreement.n_features",
                "params:deseq2_clustering_agreement.r_project",
                "params:deseq2_clustering_agreement.nproc",
            ]

        consensus_node = next(n for n in pipeline.nodes if len(n.outputs) == 2)
        assert list(consensus_node.inputs) == [
            "params:clustering_agreement.p_adj_threshold",
            "params:clustering_agreement.lfc_gap_threshold",
            "feature_significance__part1",
            "feature_significance__part2",
        ]
        assert list(consensus_node.outputs) == [
            "clustering_agreement_features_global",
            "clustering_agreement_features_local",
        ]

    def test_empty_partition_list_produces_only_aggregate_nodes(self) -> None:
        pipeline = create_pipeline([])
        assert len(pipeline.nodes) == 1
        consensus_node = pipeline.nodes[0]
        assert list(consensus_node.outputs) == [
            "clustering_agreement_features_global",
            "clustering_agreement_features_local",
        ]
        assert list(consensus_node.inputs) == [
            "params:clustering_agreement.p_adj_threshold",
            "params:clustering_agreement.lfc_gap_threshold",
        ]
