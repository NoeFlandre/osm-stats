"""Command-line interface for the OSM taginfo analysis pipeline.

Run with::

    python -m src.cli                    # default analysis (summary + building audit)
    python -m src.cli --build-cache      # materialize the thresholded tag_features cache
    python -m src.cli --summary-only     # only write global_summary.csv

Override the database location with the ``OSM_DB_PATH`` environment variable.
The cache is written to ``output/tag_features.sqlite`` by default.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.config import (
    DEFAULT_CACHE_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_THRESHOLD,
    resolve_db_path,
)
from src.core.storage.cache import read_cache_df
from src.core.db.database import OSMDatabase
from src.core.pipeline.pipeline import build_feature_cache, run
from src.core.pipeline.progress import make_progress_reporter


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OSM taginfo analysis pipeline")
    parser.add_argument(
        "--build-cache",
        action="store_true",
        help="Materialize the thresholded tag_features.sqlite cache and exit.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help="Where to write the cache (default: output/tag_features.sqlite).",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_THRESHOLD,
        help="Minimum count_all for the cache (default: 500).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
        help="Rows per batch when building the cache (default: 50000).",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only write global_summary.csv; skip the building audit.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to write CSV outputs (default: output/).",
    )
    return parser.parse_args(argv)


def _cmd_build_cache(db: OSMDatabase, args: argparse.Namespace) -> int:
    progress, t0 = make_progress_reporter(args.threshold, args.batch_size)
    build_feature_cache(
        db,
        args.cache_path,
        min_count=args.threshold,
        batch_size=args.batch_size,
        progress=progress,
    )
    print(f"  -> done in {time.time() - t0:.1f}s")
    df = read_cache_df(args.cache_path, min_count=args.threshold)
    print(f"Cache verified: {len(df):,} rows, columns={list(df.columns)}")
    return 0


def _cmd_summary_only(db: OSMDatabase, args: argparse.Namespace) -> int:
    run(db, args.output_dir, summary_only=True)
    print(f"Summary written to {args.output_dir / 'global_summary.csv'}")
    return 0


def _cmd_audit(db: OSMDatabase, args: argparse.Namespace) -> int:
    print("Auditing 'building' key...")
    df_building_audit = run(db, args.output_dir)
    print("\n--- Top 20 Building Tags ---")
    print(df_building_audit.head(20).to_string(index=False))
    print("----------------------------\n")
    print("Analysis complete! Results saved to the 'output' directory.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("Initializing OSM Database analysis...")
    db_path = resolve_db_path()
    try:
        db = OSMDatabase(str(db_path))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    if args.build_cache:
        return _cmd_build_cache(db, args)
    if args.summary_only:
        return _cmd_summary_only(db, args)
    return _cmd_audit(db, args)


if __name__ == "__main__":
    raise SystemExit(main())
