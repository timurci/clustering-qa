"""Tests for the feature partition CLI."""

# ruff: noqa: D103

from pathlib import Path

import polars as pl
import pytest

from clustering_qa.tools.partition import main, partition_features


def _make_merged(tmp_path: Path) -> tuple[Path, Path]:
    features = pl.DataFrame(
        {
            "specimen_id": ["s1", "s2", "s3", "s4", "s5", "s6"],
            "ENSG0001": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "ENSG0002": [1, 2, 3, 4, 5, 6],
        }
    )
    metadata = pl.DataFrame(
        {
            "specimen_id": ["s1", "s2", "s3", "s4", "s5", "s6"],
            "partition": ["pA", "pA", "pB", "pB", "pB", "pA"],
        }
    )
    feat_path = tmp_path / "merged.parquet"
    meta_path = tmp_path / "meta.parquet"
    features.write_parquet(str(feat_path))
    metadata.write_parquet(str(meta_path))
    return feat_path, meta_path


def test_partition_creates_per_partition_files(tmp_path: Path) -> None:
    feat_path, meta_path = _make_merged(tmp_path)
    out_dir = tmp_path / "out"
    counts = partition_features(
        feat_path, meta_path, out_dir, "specimen_id", "partition"
    )
    assert counts == {"pA": 3, "pB": 3}
    assert (out_dir / "pA.parquet").exists()
    assert (out_dir / "pB.parquet").exists()
    p_a = pl.read_parquet(str(out_dir / "pA.parquet"))
    assert set(p_a["specimen_id"].to_list()) == {"s1", "s2", "s6"}
    p_b = pl.read_parquet(str(out_dir / "pB.parquet"))
    assert set(p_b["specimen_id"].to_list()) == {"s3", "s4", "s5"}
    assert p_a.columns == ["specimen_id", "ENSG0001", "ENSG0002"]


def test_dry_run_creates_nothing(tmp_path: Path) -> None:
    feat_path, meta_path = _make_merged(tmp_path)
    out_dir = tmp_path / "out"
    counts = partition_features(
        feat_path, meta_path, out_dir, "specimen_id", "partition", dry_run=True
    )
    assert counts == {"pA": 3, "pB": 3}
    assert not (out_dir / "pA.parquet").exists()
    assert not (out_dir / "pB.parquet").exists()


def test_overwrite_guard(tmp_path: Path) -> None:
    feat_path, meta_path = _make_merged(tmp_path)
    out_dir = tmp_path / "out"
    partition_features(feat_path, meta_path, out_dir, "specimen_id", "partition")
    with pytest.raises(FileExistsError, match="already exists"):
        partition_features(
            feat_path, meta_path, out_dir, "specimen_id", "partition", overwrite=False
        )
    counts = partition_features(
        feat_path, meta_path, out_dir, "specimen_id", "partition", overwrite=True
    )
    assert counts == {"pA": 3, "pB": 3}


def test_custom_id_and_partition_columns(tmp_path: Path) -> None:
    features = pl.DataFrame({"my_id": ["a", "b", "c"], "f1": [1, 2, 3]})
    metadata = pl.DataFrame({"my_id": ["a", "b", "c"], "my_part": ["x", "x", "y"]})
    feat_path = tmp_path / "f.parquet"
    meta_path = tmp_path / "m.parquet"
    features.write_parquet(str(feat_path))
    metadata.write_parquet(str(meta_path))
    out_dir = tmp_path / "out"
    counts = partition_features(feat_path, meta_path, out_dir, "my_id", "my_part")
    assert counts == {"x": 2, "y": 1}


def test_duplicate_id_in_metadata_raises(tmp_path: Path) -> None:
    features = pl.DataFrame({"specimen_id": ["s1", "s2"], "f": [1, 2]})
    metadata = pl.DataFrame({"specimen_id": ["s1", "s1"], "partition": ["pA", "pA"]})
    feat_path = tmp_path / "f.parquet"
    meta_path = tmp_path / "m.parquet"
    features.write_parquet(str(feat_path))
    metadata.write_parquet(str(meta_path))
    with pytest.raises(ValueError, match="duplicated"):
        partition_features(
            feat_path, meta_path, tmp_path / "out", "specimen_id", "partition"
        )


def test_missing_column_raises(tmp_path: Path) -> None:
    features = pl.DataFrame({"specimen_id": ["s1"], "f": [1]})
    metadata = pl.DataFrame({"specimen_id": ["s1"], "partition": ["pA"]})
    feat_path = tmp_path / "f.parquet"
    meta_path = tmp_path / "m.parquet"
    features.write_parquet(str(feat_path))
    metadata.write_parquet(str(meta_path))
    with pytest.raises(ValueError, match="missing required"):
        partition_features(
            feat_path, meta_path, tmp_path / "out", "bad_id", "partition"
        )
    with pytest.raises(ValueError, match="missing required"):
        partition_features(
            feat_path, meta_path, tmp_path / "out", "specimen_id", "bad_part"
        )


def test_empty_partition_raises(tmp_path: Path) -> None:
    features = pl.DataFrame({"specimen_id": ["s1"], "f": [1]})
    metadata = pl.DataFrame({"specimen_id": ["s1", "s2"], "partition": ["pA", "pB"]})
    feat_path = tmp_path / "f.parquet"
    meta_path = tmp_path / "m.parquet"
    features.write_parquet(str(feat_path))
    metadata.write_parquet(str(meta_path))
    with pytest.raises(ValueError, match="would be empty"):
        partition_features(
            feat_path, meta_path, tmp_path / "out", "specimen_id", "partition"
        )


def test_non_parquet_suffix_raises(tmp_path: Path) -> None:
    csv_path = tmp_path / "f.csv"
    csv_path.write_text("a,b\n1,2\n")
    meta_path = tmp_path / "m.parquet"
    pl.DataFrame({"specimen_id": ["s1"], "partition": ["pA"]}).write_parquet(
        str(meta_path)
    )
    with pytest.raises(ValueError, match=r"expected \.parquet"):
        partition_features(
            csv_path, meta_path, tmp_path / "out", "specimen_id", "partition"
        )


def test_cli_main(tmp_path: Path) -> None:
    feat_path, meta_path = _make_merged(tmp_path)
    out_dir = tmp_path / "out"

    rc = main(
        [
            "--features",
            str(feat_path),
            "--metadata",
            str(meta_path),
            "--out",
            str(out_dir),
        ]
    )
    assert rc == 0
    assert (out_dir / "pA.parquet").exists()

    rc = main(
        [
            "--features",
            str(feat_path),
            "--metadata",
            str(meta_path),
            "--out",
            str(out_dir),
        ]
    )
    assert rc == 1

    rc = main(
        [
            "--features",
            str(feat_path),
            "--metadata",
            str(meta_path),
            "--out",
            str(out_dir),
            "--overwrite",
        ]
    )
    assert rc == 0
