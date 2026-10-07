"""Unit tests for the deseq2_clustering_agreement pipeline nodes."""

import math
import shutil
import subprocess
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
import pytest

from clustering_qa.datasets.feature_table import FeatureTable, SampleLabels
from clustering_qa.pipelines.deseq2_clustering_agreement import nodes
from clustering_qa.pipelines.deseq2_clustering_agreement.nodes import (
    deseq2_lrt_per_feature,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_R_PROJECT = _REPO_ROOT / "r" / "deseq2_clustering_agreement"
_Runner = Callable[[list[str], str], None]


def _counts_table(**genes: list[Any]) -> FeatureTable:
    """Builds a FeatureTable from ``gene -> per-sample values`` keyword args."""
    n = len(next(iter(genes.values())))
    data: dict[str, list[Any]] = {"sample_id": list(range(n))}
    data.update(genes)
    return FeatureTable(data=pl.DataFrame(data))


def _make_labels(n: int, groups: int = 2) -> SampleLabels:
    """Round-robin label assignment for ``n`` samples."""
    return SampleLabels(
        data=pl.DataFrame(
            {"sample_id": list(range(n)), "label": [str(i % groups) for i in range(n)]}
        )
    )


def _results_frame(features: list[str]) -> pl.DataFrame:
    """Builds a valid DESeq2 result table for ``features``."""
    return pl.DataFrame(
        {
            "feature": features,
            "p_value": [0.01] * len(features),
            "p_adj": [0.02] * len(features),
            "min_gap": [0.5] * len(features),
            "max_gap": [1.5] * len(features),
        }
    )


def _fake_runner(results: pl.DataFrame) -> _Runner:
    """Returns a stand-in for ``_run_deseq2_lrt`` writing ``results``."""

    def run(command: list[str], _r_project: str, _nproc: int | None = None) -> None:
        results.write_csv(Path(command[command.index("--out") + 1]))

    return run


class _SchemaCounter:
    """Frame proxy counting how often a frame's schema is resolved."""

    def __init__(self, frame: pl.DataFrame) -> None:
        self._frame = frame
        self.resolutions = 0

    @property
    def schema(self) -> dict[str, Any]:
        self.resolutions += 1
        return self._frame.schema

    def collect_schema(self) -> pl.Schema:
        self.resolutions += 1
        return self._frame.collect_schema()

    def __getattr__(self, name: str) -> object:
        return getattr(self._frame, name)


class TestDeseq2LrtPerFeature:
    """Tests for the ``deseq2_lrt_per_feature`` node function."""

    def test_returns_one_row_per_feature(self, monkeypatch: pytest.MonkeyPatch) -> None:
        results = _results_frame(["g1", "g2", "g3"])
        monkeypatch.setattr(nodes, "_run_deseq2_lrt", _fake_runner(results))
        features = _counts_table(g1=[10] * 12, g2=[20] * 12, g3=[30] * 12)
        result = deseq2_lrt_per_feature(features, _make_labels(12), 0.05)
        assert list(result.data.columns) == [
            "feature",
            "p_value",
            "p_adj",
            "min_gap",
            "max_gap",
        ]
        assert result.data["feature"].to_list() == ["g1", "g2", "g3"]
        assert result.data["p_value"].to_list() == [0.01, 0.01, 0.01]

    def test_parses_nan_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        results = _results_frame(["g1", "g2"]).with_columns(
            pl.Series("p_value", [float("nan"), 0.01]),
            pl.Series("min_gap", [float("nan"), 0.5]),
        )
        monkeypatch.setattr(nodes, "_run_deseq2_lrt", _fake_runner(results))
        features = _counts_table(g1=[10] * 12, g2=[20] * 12)
        result = deseq2_lrt_per_feature(features, _make_labels(12), 0.05)
        assert math.isnan(result.data["p_value"][0])
        assert math.isnan(result.data["min_gap"][0])
        assert result.data["p_value"][1] == 0.01

    def test_n_features_limits_to_first_columns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        results = _results_frame(["g1", "g2"])
        monkeypatch.setattr(nodes, "_run_deseq2_lrt", _fake_runner(results))
        features = _counts_table(g1=[10] * 12, g2=[20] * 12, g3=[30] * 12)
        with pytest.warns(UserWarning, match="n_features=2 limits testing"):
            result = deseq2_lrt_per_feature(
                features, _make_labels(12), 0.05, n_features=2
            )
        assert result.data["feature"].to_list() == ["g1", "g2"]

    def test_n_features_none_tests_all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        results = _results_frame(["g1", "g2", "g3"])
        monkeypatch.setattr(nodes, "_run_deseq2_lrt", _fake_runner(results))
        features = _counts_table(g1=[10] * 12, g2=[20] * 12, g3=[30] * 12)
        result = deseq2_lrt_per_feature(
            features, _make_labels(12), 0.05, n_features=None
        )
        assert result.data["feature"].to_list() == ["g1", "g2", "g3"]

    def test_n_features_larger_than_total_does_not_warn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        results = _results_frame(["g1", "g2"])
        monkeypatch.setattr(nodes, "_run_deseq2_lrt", _fake_runner(results))
        features = _counts_table(g1=[10] * 12, g2=[20] * 12)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = deseq2_lrt_per_feature(
                features, _make_labels(12), 0.05, n_features=10
            )
        assert result.data["feature"].to_list() == ["g1", "g2"]
        assert not any("n_features" in str(w.message) for w in caught)

    def test_single_group_skips_script_and_returns_nan(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _no_run(*_args: object, **_kwargs: object) -> None:
            msg = "the DESeq2 script must not run"
            raise AssertionError(msg)

        monkeypatch.setattr(nodes, "_run_deseq2_lrt", _no_run)
        features = _counts_table(g1=[10] * 12, g2=[20] * 12)
        labels = SampleLabels(
            data=pl.DataFrame({"sample_id": list(range(12)), "label": ["a"] * 12})
        )
        result = deseq2_lrt_per_feature(features, labels, 0.05)
        assert result.data["feature"].to_list() == ["g1", "g2"]
        for column in ("p_value", "p_adj", "min_gap", "max_gap"):
            assert all(math.isnan(value) for value in result.data[column].to_list())

    def test_no_feature_columns_returns_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _no_run(*_args: object, **_kwargs: object) -> None:
            msg = "the DESeq2 script must not run"
            raise AssertionError(msg)

        monkeypatch.setattr(nodes, "_run_deseq2_lrt", _no_run)
        features = FeatureTable(data=pl.DataFrame({"sample_id": list(range(12))}))
        result = deseq2_lrt_per_feature(features, _make_labels(12), 0.05)
        assert result.data.is_empty()

    def test_empty_join_raises(self) -> None:
        features = _counts_table(g1=[10] * 10)
        labels = SampleLabels(
            data=pl.DataFrame(
                {"sample_id": list(range(100, 110)), "label": ["a"] * 5 + ["b"] * 5}
            )
        )
        with pytest.raises(ValueError, match="inner join"):
            deseq2_lrt_per_feature(features, labels, 0.05)

    def test_negative_n_features_raises(self) -> None:
        features = _counts_table(g1=[10] * 12)
        with pytest.raises(ValueError, match="n_features must be positive"):
            deseq2_lrt_per_feature(features, _make_labels(12), 0.05, n_features=-1)

    def test_zero_n_features_raises(self) -> None:
        features = _counts_table(g1=[10] * 12)
        with pytest.raises(ValueError, match="n_features must be positive"):
            deseq2_lrt_per_feature(features, _make_labels(12), 0.05, n_features=0)

    @pytest.mark.parametrize("threshold", [-0.1, 1.5])
    def test_p_adj_threshold_out_of_range_raises(self, threshold: float) -> None:
        features = _counts_table(g1=[10] * 12)
        with pytest.raises(ValueError, match="p_adj_threshold must be in"):
            deseq2_lrt_per_feature(features, _make_labels(12), threshold)

    def test_non_integer_counts_raise(self) -> None:
        features = _counts_table(g1=[10.5] * 12)
        with pytest.raises(ValueError, match="non-negative integer counts"):
            deseq2_lrt_per_feature(features, _make_labels(12), 0.05)

    def test_negative_counts_raise(self) -> None:
        features = _counts_table(g1=[-5] * 12)
        with pytest.raises(ValueError, match="non-negative integer counts"):
            deseq2_lrt_per_feature(features, _make_labels(12), 0.05)

    def test_non_finite_counts_raise(self) -> None:
        features = _counts_table(g1=[float("nan")] * 12)
        with pytest.raises(ValueError, match="non-negative integer counts"):
            deseq2_lrt_per_feature(features, _make_labels(12), 0.05)

    def test_non_numeric_counts_raise(self) -> None:
        features = _counts_table(g1=["x"] * 12)
        with pytest.raises(ValueError, match="non-negative integer counts"):
            deseq2_lrt_per_feature(features, _make_labels(12), 0.05)

    def test_non_zero_nproc_raises(self) -> None:
        features = _counts_table(g1=[10] * 12)
        with pytest.raises(ValueError, match="nproc must be positive"):
            deseq2_lrt_per_feature(features, _make_labels(12), 0.05, nproc=0)

    def test_unexpected_result_columns_raise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            nodes, "_run_deseq2_lrt", _fake_runner(pl.DataFrame({"gene": ["g1"]}))
        )
        features = _counts_table(g1=[10] * 12)
        with pytest.raises(RuntimeError, match="unexpected DESeq2 result columns"):
            deseq2_lrt_per_feature(features, _make_labels(12), 0.05)

    def test_wrong_result_row_count_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        results = _results_frame(["g1"])
        monkeypatch.setattr(nodes, "_run_deseq2_lrt", _fake_runner(results))
        features = _counts_table(g1=[10] * 12, g2=[20] * 12)
        with pytest.raises(RuntimeError, match="1 rows for 2 tested features"):
            deseq2_lrt_per_feature(features, _make_labels(12), 0.05)

    def test_misordered_result_rows_raise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A feature-order mismatch would be attributed to the wrong genes."""
        results = _results_frame(["g2", "g1"])
        monkeypatch.setattr(nodes, "_run_deseq2_lrt", _fake_runner(results))
        features = _counts_table(g1=[10] * 12, g2=[20] * 12)
        with pytest.raises(RuntimeError, match="do not match the tested features"):
            deseq2_lrt_per_feature(features, _make_labels(12), 0.05)


class TestRawCountMatrix:
    """Tests for the ``_raw_count_matrix`` helper."""

    def test_schema_is_resolved_once(self) -> None:
        """Regression: resolving the schema per column is quadratic.

        ``DataFrame.schema`` rebuilds a dict of every column (~19 ms for the
        raw tables' 67k columns), which made this validation take ~21
        minutes per cohort instead of milliseconds.
        """
        frame = _SchemaCounter(
            pl.DataFrame({"sample_id": [0, 1], "g1": [1, 2], "g2": [3, 4]})
        )
        result = nodes._raw_count_matrix(cast("pl.DataFrame", frame), ["g1", "g2"])
        assert frame.resolutions == 1
        assert result.columns == ["sample_id", "g1", "g2"]
        assert result["g1"].dtype == pl.Int64


class TestDeseq2Command:
    """Tests for the ``_deseq2_command`` argv builder."""

    def test_flags_match_the_request(self, tmp_path: Path) -> None:
        counts, labels, out = (
            tmp_path / name for name in ("counts.csv", "labels.csv", "out.csv")
        )
        command = nodes._deseq2_command(counts, labels, out, 0.05, 6)
        assert command[:3] == ["uvr", "run", "run_deseq2_lrt.R"]
        assert command[-2:] == ["--nproc", "6"]
        assert command[command.index("--counts") + 1] == str(counts)
        assert command[command.index("--labels") + 1] == str(labels)
        assert command[command.index("--out") + 1] == str(out)
        assert command[command.index("--alpha") + 1] == "0.05"

    def test_none_nproc_is_not_passed(self, tmp_path: Path) -> None:
        """An empty flag would make the R parser reject the argument."""
        command = nodes._deseq2_command(
            tmp_path / "counts.csv",
            tmp_path / "labels.csv",
            tmp_path / "out.csv",
            0.05,
            None,
        )
        assert "--nproc" not in command


class TestRunDeseq2Lrt:
    """Tests for the ``_run_deseq2_lrt`` subprocess wrapper."""

    def test_nonzero_exit_raises_with_stderr(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_run(
            command: list[str], **_kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(command, 1, "", "boom: R exploded")

        monkeypatch.setattr(nodes.subprocess, "run", fake_run)
        with pytest.raises(RuntimeError, match="exit code 1") as exc_info:
            nodes._run_deseq2_lrt(["uvr", "run"], "r/deseq2_clustering_agreement")
        assert "boom: R exploded" in str(exc_info.value)

    def test_missing_uvr_raises_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_run(_command: list[str], **_kwargs: object) -> None:
            msg = "uvr"
            raise FileNotFoundError(msg)

        monkeypatch.setattr(nodes.subprocess, "run", fake_run)
        with pytest.raises(RuntimeError, match="uvr executable not found"):
            nodes._run_deseq2_lrt(["uvr", "run"], "r/deseq2_clustering_agreement")

    def test_missing_r_project_is_not_reported_as_missing_uvr(self) -> None:
        """A bad ``r_project`` and a missing ``uvr`` raise the same error."""
        missing = str(_REPO_ROOT / "r" / "no_such_project")
        with pytest.raises(RuntimeError, match="does not exist"):
            nodes._run_deseq2_lrt(["uvr", "run"], missing)

    @pytest.mark.parametrize(("nproc", "capped"), [(None, True), (4, True), (1, False)])
    def test_math_libraries_are_single_threaded_when_forking(
        self, monkeypatch: pytest.MonkeyPatch, nproc: int | None, *, capped: bool
    ) -> None:
        """Forked workers deadlock on inherited BLAS/OpenMP thread pools."""
        seen: dict[str, str] = {}

        def fake_run(
            command: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            env = cast("Mapping[str, str]", kwargs["env"])
            seen.update(env)
            return subprocess.CompletedProcess(command, 0, "", "")

        monkeypatch.setattr(nodes.subprocess, "run", fake_run)
        nodes._run_deseq2_lrt(["uvr", "run"], "r/deseq2_clustering_agreement", nproc)
        assert (seen.get("OMP_NUM_THREADS") == "1") is capped
        assert (seen.get("OPENBLAS_NUM_THREADS") == "1") is capped


@pytest.mark.skipif(
    shutil.which("uvr") is None or not (_R_PROJECT / "uvr.lock").exists(),
    reason="uvr or the R project is not set up",
)
class TestDeseq2LrtEndToEnd:
    """Integration tests running the real DESeq2 script via uvr."""

    def test_separable_genes_are_detected(self) -> None:
        rng = np.random.default_rng(11)
        n_per_group, n_null, n_shift = 6, 20, 5
        n = 2 * n_per_group
        counts = rng.negative_binomial(10, 0.3, size=(n, n_null + n_shift)).astype(
            np.int64
        )
        counts[n_per_group:, n_null:] += 80
        genes = [f"gene{i}" for i in range(n_null + n_shift)]
        features = FeatureTable(
            data=pl.DataFrame(counts, schema=genes).insert_column(
                0, pl.Series("sample_id", list(range(n)))
            )
        )
        labels = SampleLabels(
            data=pl.DataFrame(
                {
                    "sample_id": list(range(n)),
                    "label": ["0"] * n_per_group + ["1"] * n_per_group,
                }
            )
        )
        result = deseq2_lrt_per_feature(
            features, labels, 0.05, r_project=str(_R_PROJECT)
        )
        rows = {row["feature"]: row for row in result.data.to_dicts()}
        assert set(rows) == set(genes)
        shifted = genes[n_null:]
        for gene in shifted:
            assert rows[gene]["p_value"] < 0.01
        # Two label groups means one pair, so the two gaps coincide.
        for row in rows.values():
            assert row["min_gap"] == row["max_gap"]
        # The shifted genes are two-fold up in one group: their gap clears the
        # null genes', which sit at zero.
        assert min(rows[gene]["max_gap"] for gene in shifted) > max(
            rows[gene]["max_gap"] for gene in genes[:n_null]
        )
        for gene in shifted:
            assert rows[gene]["min_gap"] > 0.5
