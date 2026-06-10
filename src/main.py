"""Entry point for the OSM taginfo analysis pipeline.

Run with::

    python -m src.main                  # default analysis (summary + building audit)
    python -m src.main --build-cache    # materialize the thresholded tag_features cache
    python -m src.main --summary-only   # only write global_summary.csv

Override the database location with the ``OSM_DB_PATH`` environment variable.
The cache is written to ``output/tag_features.sqlite`` by default.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.config import (
    DEFAULT_CACHE_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_THRESHOLD,
    resolve_db_path,
)
from src.core.audit import OSMTagAuditor
from src.core.cache import build_cache_db, build_cache_db_streaming, read_cache_df
from src.core.database import OSMDatabase
from src.core.summary import SummaryBuilder
from src.io.exporter import DataExporter


def build_summary(db: OSMDatabase) -> pd.DataFrame:
    """Aggregate the global key/tag totals into a single tidy DataFrame."""
    return SummaryBuilder(db).build().to_dataframe()


def build_feature_cache(
    db: OSMDatabase,
    output_path: Path = DEFAULT_CACHE_PATH,
    min_count: int = DEFAULT_THRESHOLD,
    batch_size: int = 50_000,
) -> Path:
    """Threshold the tags table, standardize each row, and write a sqlite cache.

    Streams the result in batches of *batch_size* rows using a keyset on
    ``count_all``, so memory usage stays bounded even on the full 192 M
    source rows. Returns the cache path. The cache schema is::

        CREATE TABLE tag_features (
            key TEXT, value TEXT, count_all INTEGER, feature TEXT,
            PRIMARY KEY (key, value)
        );

    where ``feature`` is ``"key|value"`` (lowercased, stripped).
    """
    print(f"Streaming rows with count_all >= {min_count:,} in batches of {batch_size:,} ...")
    t0 = time.time()
    last_report = t0

    def _progress(n: int, last_count: int | None) -> None:
        nonlocal last_report
        now = time.time()
        if now - last_report >= 5.0 or n == batch_size:
            rate = n / max(now - t0, 1e-3)
            print(f"  {n:,} rows written  ({rate:,.0f} rows/s, last count_all={last_count})")
            last_report = now

    build_cache_db_streaming(
        db,
        output_path,
        min_count=min_count,
        batch_size=batch_size,
        progress=_progress,
    )
    print(f"  -> done in {time.time() - t0:.1f}s")
    return output_path


def run(
    db: OSMDatabase,
    output_dir: Path,
    summary_only: bool = False,
) -> pd.DataFrame:
    """Execute the analysis pipeline against *db* and write results to *output_dir*.

    Returns the building-audit DataFrame for inspection.
    """
    exporter = DataExporter(db, output_dir)

    summary = build_summary(db)
    summary.to_csv(output_dir / "global_summary.csv", index=False)

    if summary_only:
        return pd.DataFrame()

    auditor = OSMTagAuditor(db)
    df_building_audit = auditor.top_values("building", limit=50)
    df_building_audit.to_csv(output_dir / "audit_building.csv", index=False)
    return df_building_audit


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
        build_feature_cache(
            db,
            args.cache_path,
            min_count=args.threshold,
            batch_size=args.batch_size,
        )
        # Quick verification read.
        df = read_cache_df(args.cache_path, min_count=args.threshold)
        print(f"Cache verified: {len(df):,} rows, columns={list(df.columns)}")
        return 0

    if args.summary_only:
        run(db, args.output_dir, summary_only=True)
        print(f"Summary written to {args.output_dir / 'global_summary.csv'}")
        return 0

    print("Auditing 'building' key...")
    df_building_audit = run(db, args.output_dir)

    print("\n--- Top 20 Building Tags ---")
    print(df_building_audit.head(20).to_string(index=False))
    print("----------------------------\n")

    print("Analysis complete! Results saved to the 'output' directory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
