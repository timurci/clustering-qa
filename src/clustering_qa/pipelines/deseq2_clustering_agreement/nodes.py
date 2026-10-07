"""Nodes for the DESeq2 clustering agreement pipeline."""

import os
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np
import polars as pl

from clustering_qa.datasets.feature_significance import FeatureSignificance
from clustering_qa.datasets.feature_table import FeatureTable, SampleLabels

_SCRIPT_NAME = "run_deseq2_lrt.R"
DEFAULT_R_PROJECT = "r/deseq2_clustering_agreement"
_MIN_GROUPS_FOR_LRT = 2
_RESULT_COLUMNS = ["feature", "p_value", "p_adj", "min_gap", "max_gap"]
_COUNTS_REQUIREMENT = "DESeq2 requires raw non-negative integer counts"


def deseq2_lrt_per_feature(  # noqa: PLR0913 - inputs plus the test settings
    features: FeatureTable,
    labels: SampleLabels,
    p_adj_threshold: float,
    n_features: int | None = None,
    r_project: str = DEFAULT_R_PROJECT,
    nproc: int | None = None,
) -> FeatureSignificance:
    """Run a DESeq2 likelihood ratio test for every feature in one partition.

    The feature table must hold raw RNA-seq counts (non-negative integers);
    they are exported to CSV together with the labels and tested by
    ``run_deseq2_lrt.R`` in the uvr-managed R project at ``r_project``.
    The script runs ``DESeq(test = "LRT", reduced = ~ 1)`` per feature,
    so the p-value asks whether the feature differs across any label
    group. Adjusted p-values come from ``DESeq2::results()``
    (Benjamini-Hochberg with independent filtering at ``p_adj_threshold``).

    The effect size is the pairwise log2 fold change between every pair of
    label levels, shrunk with ashr and reduced to a lower confidence bound
    ``|lfc| - z * posterior SD``. ``min_gap`` is the smallest bound over the
    pairs (large only when every cluster pair is separated) and ``max_gap``
    the largest (large when at least one pair is separated).

    Features that cannot be tested (single group after the inner join,
    constant counts, or genes DESeq2 drops) receive ``NaN`` p-values,
    ``NaN`` adjusted p-values, and ``NaN`` gaps.

    Args:
        features: Raw count table whose non-id columns are tested.
        labels: Per-sample labels that define the strata for the test.
        p_adj_threshold: FDR target passed to ``DESeq2::results()`` for
            independent filtering.
        n_features: Optional limit on the number of features tested; the
            first ``n_features`` columns are used. ``None`` tests all
            features.
        r_project: Path to the uvr project holding the DESeq2 script,
            relative to the Kedro project root (run ``kedro`` from there).
        nproc: Number of workers for the per-gene model fits; ``None``
            lets the R script pick one fewer than the detected cores,
            capped at 8.

    Returns:
        A :class:`FeatureSignificance` with columns ``feature``, ``p_value``,
        ``p_adj``, ``min_gap``, and ``max_gap``.

    Raises:
        ValueError: If the inputs are empty after the join, ``n_features``
            or ``nproc`` is not positive, or the counts are not raw
            non-negative integers.
        RuntimeError: If the DESeq2 script fails or returns a malformed
            result table.
    """
    if n_features is not None and n_features <= 0:
        msg = f"n_features must be positive, got {n_features}"
        raise ValueError(msg)
    if nproc is not None and nproc <= 0:
        msg = f"nproc must be positive, got {nproc}"
        raise ValueError(msg)
    if not 0.0 <= p_adj_threshold <= 1.0:
        msg = f"p_adj_threshold must be in [0, 1], got {p_adj_threshold}"
        raise ValueError(msg)

    joined = features.data.join(labels.data, on="sample_id", how="inner")
    if joined.is_empty():
        msg = "inner join of features and labels produced no rows"
        raise ValueError(msg)

    feature_columns = [col for col in features.data.columns if col != "sample_id"]
    if n_features is not None and len(feature_columns) > n_features:
        truncated = feature_columns[:n_features]
        warnings.warn(
            f"n_features={n_features} limits testing to the first "
            f"{len(truncated)} of {len(feature_columns)} features",
            stacklevel=2,
        )
        feature_columns = truncated

    if not feature_columns:
        return _nan_significance(feature_columns)

    counts = _raw_count_matrix(joined, feature_columns)
    if joined["label"].n_unique() < _MIN_GROUPS_FOR_LRT:
        return _nan_significance(feature_columns)

    with tempfile.TemporaryDirectory() as tmpdir:
        counts_path = Path(tmpdir) / "counts.csv"
        labels_path = Path(tmpdir) / "labels.csv"
        out_path = Path(tmpdir) / "results.csv"
        counts.write_csv(counts_path)
        joined.select(["sample_id", "label"]).write_csv(labels_path)
        _run_deseq2_lrt(
            _deseq2_command(counts_path, labels_path, out_path, p_adj_threshold, nproc),
            r_project,
            nproc,
        )
        results = pl.read_csv(out_path)

    return FeatureSignificance(data=_parse_results(results, feature_columns))


def _nan_significance(feature_columns: list[str]) -> FeatureSignificance:
    """Return a significance table with one all-NaN row per feature."""
    return FeatureSignificance(
        data=pl.DataFrame(
            {
                "feature": feature_columns,
                "p_value": np.full(len(feature_columns), np.nan),
                "p_adj": np.full(len(feature_columns), np.nan),
                "min_gap": np.full(len(feature_columns), np.nan),
                "max_gap": np.full(len(feature_columns), np.nan),
            }
        )
    )


def _raw_count_matrix(joined: pl.DataFrame, feature_columns: list[str]) -> pl.DataFrame:
    """Validate raw counts and return them as an integer sample-by-gene table."""
    # Resolve the schema once: ``DataFrame.schema`` rebuilds a dict of every
    # column (~19 ms for the 67k-column raw tables), so looking it up per
    # feature made this check quadratic — 21 minutes per cohort.
    schema = joined.schema
    numeric_columns = [col for col in feature_columns if schema[col].is_numeric()]
    if len(numeric_columns) != len(feature_columns):
        msg = f"{_COUNTS_REQUIREMENT}; got non-numeric feature columns"
        raise ValueError(msg)
    values = joined.select(feature_columns).cast(pl.Float64).to_numpy()
    if not np.isfinite(values).all():
        msg = f"{_COUNTS_REQUIREMENT}; got non-finite values"
        raise ValueError(msg)
    if (values < 0).any():
        msg = f"{_COUNTS_REQUIREMENT}; got negative values"
        raise ValueError(msg)
    if not np.equal(values, np.floor(values)).all():
        msg = f"{_COUNTS_REQUIREMENT}; got non-integer values"
        raise ValueError(msg)
    return joined.select(["sample_id", *feature_columns]).cast(
        dict.fromkeys(feature_columns, pl.Int64)
    )


def _deseq2_command(
    counts_path: Path,
    labels_path: Path,
    out_path: Path,
    alpha: float,
    nproc: int | None,
) -> list[str]:
    """Build the ``uvr run`` command for the DESeq2 LRT script.

    Args:
        counts_path: Sample-by-gene raw count CSV written for the script.
        labels_path: Sample-to-label CSV written for the script.
        out_path: Destination of the script's per-gene result table.
        alpha: FDR target passed to ``DESeq2::results()``.
        nproc: Workers for the per-gene fits; ``None`` leaves the choice to
            the script.

    Returns:
        The argv to run, without the uvr project working directory.
    """
    command = [
        "uvr",
        "run",
        _SCRIPT_NAME,
        "--",
        "--counts",
        str(counts_path),
        "--labels",
        str(labels_path),
        "--out",
        str(out_path),
        "--alpha",
        str(alpha),
    ]
    if nproc is not None:
        command += ["--nproc", str(nproc)]
    return command


def _run_deseq2_lrt(
    command: list[str], r_project: str, nproc: int | None = None
) -> None:
    """Run a DESeq2 LRT command in the uvr R project.

    The R script forks one process per worker, and forking an R process whose
    BLAS/OpenMP runtime has started thread pools deadlocks the children (they
    block on inherited locks and the run never finishes), so a multi-worker
    run is started with single-threaded math libraries. One worker or an
    explicit ``nproc=1`` keeps the shared libraries' own threading.

    Args:
        command: Argv built by :func:`_deseq2_command`.
        r_project: Path to the uvr project holding the script, used as the
            working directory.
        nproc: Worker count the command requests; ``None`` means the script
            picks it (more than one on any multi-core machine).

    Raises:
        RuntimeError: If the R project directory is missing, ``uvr`` is not
            installed, or the script exits non-zero.
    """
    if not Path(r_project).is_dir():
        msg = (
            f"uvr project directory {r_project!r} does not exist; run kedro "
            "from the project root or point "
            "deseq2_clustering_agreement.r_project at the uvr project"
        )
        raise RuntimeError(msg)
    env = os.environ.copy()
    if nproc != 1:
        env.update(
            dict.fromkeys(
                (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "GOTO_NUM_THREADS",
                ),
                "1",
            )
        )
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            command,
            cwd=r_project,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
    except FileNotFoundError as exc:
        msg = (
            "uvr executable not found; install uvr and run `uvr sync` in "
            f"{r_project} to set up the DESeq2 environment"
        )
        raise RuntimeError(msg) from exc
    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip()
        msg = (
            f"DESeq2 LRT script failed with exit code {completed.returncode}:\n"
            f"{details}"
        )
        raise RuntimeError(msg)


def _parse_results(results: pl.DataFrame, feature_columns: list[str]) -> pl.DataFrame:
    """Validate and coerce the DESeq2 result table to a significance table.

    The script is expected to emit one row per tested feature in the order
    the columns were exported, so the feature names are checked as well as
    the row count; a misordered table would otherwise be silently
    attributed to the wrong features.

    Raises:
        RuntimeError: If the table does not have the expected schema, row
            count, or feature names.
    """
    if set(results.columns) != set(_RESULT_COLUMNS):
        msg = f"unexpected DESeq2 result columns: {results.columns}"
        raise RuntimeError(msg)
    if results.height != len(feature_columns):
        msg = (
            f"DESeq2 returned {results.height} rows for "
            f"{len(feature_columns)} tested features"
        )
        raise RuntimeError(msg)
    if results["feature"].to_list() != feature_columns:
        msg = "DESeq2 result rows do not match the tested features in order"
        raise RuntimeError(msg)
    return results.select(_RESULT_COLUMNS).cast(
        {
            "p_value": pl.Float64,
            "p_adj": pl.Float64,
            "min_gap": pl.Float64,
            "max_gap": pl.Float64,
        }
    )
