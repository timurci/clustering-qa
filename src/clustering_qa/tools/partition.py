"""Partition a merged feature table into per-partition parquet files."""

import argparse
import sys
from pathlib import Path

import polars as pl


def _load_parquet(path: Path) -> pl.DataFrame:
    if path.suffix != ".parquet":
        msg = f"expected .parquet file, got {path}"
        raise ValueError(msg)
    if not path.exists():
        msg = f"file not found: {path}"
        raise FileNotFoundError(msg)
    return pl.read_parquet(str(path))


def _validate_columns(df: pl.DataFrame, path: Path, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        msg = f"{path} is missing required column(s): {', '.join(missing)}; available: {df.columns}"  # noqa: E501
        raise ValueError(msg)


def partition_features(  # noqa: C901, PLR0913
    features_path: Path,
    metadata_path: Path,
    out_dir: Path,
    id_column: str,
    partition_column: str,
    *,
    overwrite: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
) -> dict[str, int]:
    """Partition merged features into per-partition parquet files."""
    features = _load_parquet(features_path)
    metadata = _load_parquet(metadata_path)

    _validate_columns(features, features_path, [id_column])
    _validate_columns(metadata, metadata_path, [id_column, partition_column])

    if metadata[id_column].is_duplicated().any():
        dups = (
            metadata.filter(pl.col(id_column).is_duplicated())[id_column]
            .unique()
            .to_list()
        )
        msg = f"metadata {metadata_path} has duplicated {id_column} values: {dups[:5]}"
        raise ValueError(msg)

    if metadata[partition_column].is_null().any():
        msg = f"metadata {metadata_path} has null values in {partition_column}"
        raise ValueError(msg)

    partitions = metadata[partition_column].unique().sort().to_list()
    if not partitions:
        msg = f"metadata {metadata_path} has no partitions in {partition_column}"
        raise ValueError(msg)

    feature_ids = set(features[id_column].to_list())
    meta_ids = set(metadata[id_column].to_list())
    only_in_meta = meta_ids - feature_ids
    only_in_features = feature_ids - meta_ids
    if verbose:
        if only_in_meta:
            print(  # noqa: T201
                f"warning: {len(only_in_meta)} ids in metadata not in features",
                file=sys.stderr,
            )
        if only_in_features:
            print(  # noqa: T201
                f"warning: {len(only_in_features)} ids in features not in metadata",
                file=sys.stderr,
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    for pid in partitions:
        pid_ids = metadata.filter(pl.col(partition_column) == pid)[id_column].to_list()
        part_df = features.filter(pl.col(id_column).is_in(pid_ids))
        n = part_df.height
        if n == 0:
            msg = f"partition {pid!r} would be empty (no features for its {len(pid_ids)} ids)"  # noqa: E501
            raise ValueError(msg)
        counts[str(pid)] = n
        out_path = out_dir / f"{pid}.parquet"
        if out_path.exists() and not overwrite:
            msg = f"{out_path} already exists (use --overwrite to replace)"
            raise FileExistsError(msg)
        if verbose or dry_run:
            print(  # noqa: T201
                f"{'[dry-run] ' if dry_run else ''}{pid}: {n} rows -> {out_path}",
                file=sys.stderr,
            )
        if not dry_run:
            part_df.write_parquet(
                str(out_path), compression="zstd", compression_level=5
            )

    return counts


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Partition a merged feature parquet into per-partition files using metadata.",  # noqa: E501
    )
    parser.add_argument(
        "--features", required=True, type=Path, help="path to merged features parquet"
    )
    parser.add_argument(
        "--metadata", required=True, type=Path, help="path to metadata parquet"
    )
    parser.add_argument(
        "--id-column",
        default="specimen_id",
        help="id column name in both files (default: specimen_id)",
    )
    parser.add_argument(
        "--partition-column",
        default="partition",
        help="partition column in metadata (default: partition)",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="output directory for {partition}.parquet files",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="overwrite existing output files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="do not write files, only print what would be done",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="print warnings and per-partition counts"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for feature partitioning."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        partition_features(
            features_path=args.features,
            metadata_path=args.metadata,
            out_dir=args.out,
            id_column=args.id_column,
            partition_column=args.partition_column,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)  # noqa: T201
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
