"""End-to-end pipeline orchestration.

This module owns the *flow* of the analysis: build the cache, build the
summary, run the building audit, write CSVs. The lower-level building blocks
(:class:`SummaryBuilder`, :class:`OSMTagAuditor`, :func:`build_cache_db_streaming`)
live in their own modules and are imported here. CLI wiring and argv parsing
live in :mod:`src.main`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from src.config import DEFAULT_CACHE_PATH, DEFAULT_THRESHOLD
from src.core.audit import OSMTagAuditor
from src.core.cache import build_cache_db_streaming
from src.core.database import OSMDatabase
from src.core.summary import SummaryBuilder


def build_summary(db: OSMDatabase) -> pd.DataFrame:
    """Aggregate the global key/tag totals into a single tidy DataFrame."""
    return SummaryBuilder(db).build().to_dataframe()


ProgressCb = Callable[[int, Optional[int]], None]  # (rows_written, last_count_all)


def build_feature_cache(
    db: OSMDatabase,
    output_path: Path = DEFAULT_CACHE_PATH,
    min_count: int = DEFAULT_THRESHOLD,
    batch_size: int = 50_000,
    progress: ProgressCb | None = None,
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

    The optional *progress* callback is invoked after each batch with
    ``(rows_written, last_count_all)``. Banner and timing are the caller's
    responsibility.
    """
    build_cache_db_streaming(
        db,
        output_path,
        min_count=min_count,
        batch_size=batch_size,
        progress=progress,
    )
    return output_path


def run(
    db: OSMDatabase,
    output_dir: Path,
    summary_only: bool = False,
) -> pd.DataFrame:
    """Execute the analysis pipeline against *db* and write results to *output_dir*.

    Returns the building-audit DataFrame for inspection.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(db)
    summary.to_csv(output_dir / "global_summary.csv", index=False)

    if summary_only:
        return pd.DataFrame()

    auditor = OSMTagAuditor(db)
    df_building_audit = auditor.top_values("building", limit=50)
    df_building_audit.to_csv(output_dir / "audit_building.csv", index=False)
    return df_building_audit
