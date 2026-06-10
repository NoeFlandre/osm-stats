"""Data loading helpers for the OSM taginfo analysis pipeline.

The first step of the pipeline is a *long-tail filter*: drop the vast majority
of low-volume tags so the downstream clustering and taxonomy stages run on a
small, dense matrix. Empirically a ``count_all >= 500`` threshold trims the
192.8 M row ``tags`` table down to a few thousand unique ``(key, value)``
pairs.

Note: the upstream ``tags`` table has an index on ``(key, count_all)`` but not
on ``count_all`` alone, so the filtered scan is O(filtered_rows) and may take
~30 s on the full planet. Pass ``limit=`` to materialize only the top N
in seconds; the downstream stages can be tuned progressively from there.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from src.core.queries import QueryBuilder
from src.io.exporter import DB


DEFAULT_MIN_COUNT = 500


def load_and_threshold(
    db: DB,
    min_count: int = DEFAULT_MIN_COUNT,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Return ``(key, value, count_all)`` rows with ``count_all >= min_count``.

    The query is parameterized; *min_count* and *limit* are never interpolated
    into the SQL string. The returned DataFrame is ordered by ``count_all``
    descending. Pass *limit* to cap the result to the top N rows.
    """
    sql, params = QueryBuilder.tags_by_min_count(min_count=min_count, limit=limit)
    return db.execute_query(sql, params)
