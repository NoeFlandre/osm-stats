"""Per-base-key element-type statistics for the env/agri study.

The source ``taginfo.sqlite`` already records, for every (key, value)
pair, how many of its occurrences live on nodes, ways and relations
(see the ``count_nodes / count_ways / count_relations`` columns of
the ``tags`` table). This module aggregates those per-base-key, so
that for a base key like ``landuse`` we can answer "are these tags
on polygons or points?" without leaving the existing data.

A base key is the part of an OSM key before the first colon. So
``landuse`` covers ``landuse=farmland``, ``landuse=forest``, ...;
``addr`` covers ``addr``, ``addr:city``, ``addr:street``, ... — see
:func:`src.core.features.base_key.parse_base_key` for the
in-Python convention. We replicate the same rule in SQL with
``substr(key, 1, instr(key, ':') - 1)`` so we never have to round-trip
the 192 M rows of the ``tags`` table through Python.

The output DataFrame carries, per base key:

* ``n_tags`` — number of distinct (key, value) pairs that rolled up
  into this base key.
* ``count_all / count_nodes / count_ways / count_relations`` — the
  element-type split, summed across those (key, value) pairs.
* ``nodes_pct / ways_pct / relations_pct`` — the same as
  percentages.
* ``is_polygon_friendly`` — ``True`` when the fraction of
  occurrences on ways+relations is at least
  :data:`POLYGON_FRIENDLY_THRESHOLD` (default: 50 %).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Union

import pandas as pd

#: A base key is considered "polygon-friendly" if at least this
#: fraction of its occurrences are on ways or relations (i.e. not on
#: nodes). The default 0.5 means "at least half of the occurrences
#: could plausibly be polygons".
POLYGON_FRIENDLY_THRESHOLD: float = 0.5

#: Canonical column order of the DataFrame returned by
#: :func:`element_type_stats`. Exposed so callers can rely on the
#: schema even when the input is empty.
ELEMENT_TYPE_STATS_COLUMNS: list[str] = [
    "base_key",
    "n_tags",
    "count_all",
    "count_nodes",
    "count_ways",
    "count_relations",
    "nodes_pct",
    "ways_pct",
    "relations_pct",
    "is_polygon_friendly",
]


def _resolve_connection(db: Union[str, Path, sqlite3.Connection]) -> sqlite3.Connection:
    """Return a sqlite3 connection for *db* (path or existing connection)."""
    if isinstance(db, sqlite3.Connection):
        return db
    return sqlite3.connect(str(db))


def element_type_stats(
    base_keys: Iterable[str],
    db: Union[str, Path, sqlite3.Connection],
) -> pd.DataFrame:
    """Compute the per-base-key element-type split from the source taginfo DB.

    Parameters
    ----------
    base_keys
        The base keys to look up. May be in any order; the returned
        DataFrame preserves the input order. Empty input returns an
        empty DataFrame with the canonical column schema.
    db
        Either a :class:`sqlite3.Connection` or a path to a taginfo
        sqlite DB that has a ``tags(key, value, count_all,
        count_nodes, count_ways, count_relations)`` table.

    Returns
    -------
    pandas.DataFrame
        One row per base key, with the columns listed in
        :data:`ELEMENT_TYPE_STATS_COLUMNS`.
    """
    base_keys_list = [str(b).strip() for b in base_keys]
    if not base_keys_list:
        return pd.DataFrame(columns=ELEMENT_TYPE_STATS_COLUMNS)

    con = _resolve_connection(db)

    # Aggregate the full tags table by the base-key expression in
    # SQL. The CASE expression mirrors src.core.features.base_key.
    # parse_base_key: take everything before the first ':', or the
    # full key if there is no colon.
    sql = """
        WITH per_bk AS (
            SELECT
                CASE
                    WHEN instr(key, ':') > 0
                        THEN substr(key, 1, instr(key, ':') - 1)
                    ELSE key
                END AS base_key,
                COUNT(*) AS n_tags,
                SUM(count_all)       AS count_all,
                SUM(count_nodes)     AS count_nodes,
                SUM(count_ways)      AS count_ways,
                SUM(count_relations) AS count_relations
            FROM tags
            GROUP BY base_key
        )
        SELECT * FROM per_bk
        WHERE base_key IN ({placeholders})
    """.format(placeholders=",".join("?" * len(base_keys_list)))

    agg = pd.read_sql_query(sql, con, params=base_keys_list)

    # Make sure every requested base key is present, even if the
    # source DB has no rows for it. Otherwise callers have to merge
    # against the input list to know which base keys were missing.
    found = set(agg["base_key"].tolist())
    missing = [b for b in base_keys_list if b not in found]
    if missing:
        zeros = pd.DataFrame(
            {
                "base_key": missing,
                "n_tags": [0] * len(missing),
                "count_all": [0] * len(missing),
                "count_nodes": [0] * len(missing),
                "count_ways": [0] * len(missing),
                "count_relations": [0] * len(missing),
            }
        )
        agg = pd.concat([agg, zeros], ignore_index=True)

    # Percentages. count_all can be 0 for a base key with no source
    # rows; guard the division.
    safe_total = agg["count_all"].where(agg["count_all"] > 0, other=1)
    agg["nodes_pct"]     = agg["count_nodes"]     / safe_total
    agg["ways_pct"]      = agg["count_ways"]      / safe_total
    agg["relations_pct"] = agg["count_relations"] / safe_total
    # Reset the percentage columns to 0 when the denominator was 0,
    # so a missing base key shows 0% rather than 0/1 = 0% (same
    # value, but explicit for readability).
    empty_mask = agg["count_all"] == 0
    for col in ("nodes_pct", "ways_pct", "relations_pct"):
        agg.loc[empty_mask, col] = 0.0

    # Polygon-friendly = at least the threshold of occurrences on
    # ways+relations (i.e. potentially polygons, not points or lines).
    ways_rel = agg["count_ways"] + agg["count_relations"]
    agg["is_polygon_friendly"] = (ways_rel / safe_total) >= POLYGON_FRIENDLY_THRESHOLD
    agg.loc[empty_mask, "is_polygon_friendly"] = False

    # Reorder columns and rows. We preserve the caller's input order.
    order_index = {b: i for i, b in enumerate(base_keys_list)}
    agg["_order"] = agg["base_key"].map(order_index)
    agg = (
        agg.sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )
    agg = agg[ELEMENT_TYPE_STATS_COLUMNS]

    return agg
