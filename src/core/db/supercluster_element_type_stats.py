"""Per-supercluster element-type statistics.

A **supercluster** is a group of HDBSCAN clusters that share the
same base key on their medoid. For a base key ``landuse``, the
supercluster is the union of every member of every cluster whose
medoid's base key is ``landuse``. This is the unit of selection in
the env/agri study: the user labels superclusters (one row per
base key in the XLSX), not individual tags.

This module computes the element-type split of each supercluster's
*actual contents* — not of every (key, value) pair globally with
that base key in the source ``taginfo.sqlite``. The two views
differ:

- the source-DB view (``element_type_stats``) includes source rows
  that ended up in noise (cluster_id = -1) and includes
  case/whitespace variants of the same standardized tag as
  separate rows;
- the supercluster view excludes noise and folds case/whitespace
  variants together (because the cluster_memberships are
  standardized, and the source-DB join uses LOWER()).

So a "landuse" supercluster whose cluster members are all on ways
will have a 100 % ways share, even if there are 1000 ``landuse``
tags in the source DB that ended up in noise with 50 % on nodes.

A base key that the user labeled but the pipeline never produced
a cluster for (no medoid has that base key) gets a zero row,
never dropped — the user can see that the supercluster is empty.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Union

import pandas as pd

from src.core.db.element_type_stats import POLYGON_FRIENDLY_THRESHOLD
from src.core.features.base_key import parse_base_key

#: Canonical column order of the DataFrame returned by

import pandas as pd

#: Canonical column order of the DataFrame returned by
#: :func:`supercluster_element_type_stats`. Exposed so callers can
#: rely on the schema even when the input is empty.
SUPERCLUSTER_STATS_COLUMNS: list[str] = [
    "base_key",
    "n_clusters",
    "n_tags",
    "count_all",
    "count_nodes",
    "count_ways",
    "count_relations",
    "pct_nodes",
    "pct_ways",
    "pct_relations",
    "is_polygon_friendly",
]


def _resolve_source_db(db: Union[str, Path, sqlite3.Connection]) -> sqlite3.Connection:
    """Return a sqlite3 connection for *db* (path or existing connection)."""
    if isinstance(db, sqlite3.Connection):
        return db
    return sqlite3.connect(str(db))


def _load_medoids(medoids: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    """Return a DataFrame with at least ``cluster_id`` and ``medoid_feature``."""
    if isinstance(medoids, pd.DataFrame):
        return medoids
    return pd.read_csv(medoids)


def _load_members(members: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    """Return a DataFrame of cluster memberships."""
    if isinstance(members, pd.DataFrame):
        return members
    return pd.read_csv(members)


def _aggregate_source_db_by_lower(
    db: sqlite3.Connection, cache_path: Union[str, Path, None] = None
) -> pd.DataFrame:
    """Group the source ``tags`` table by ``LOWER(key), LOWER(value)`` and
    sum the element-type counts.

    This folds case/whitespace variants together so the join with
    the (standardized) cluster_memberships finds the right
    element-type split. The result has one row per standardized
    (key, value) pair, with columns ``key``, ``value``,
    ``count_nodes``, ``count_ways``, ``count_relations``.

    To keep this fast on the 192 M-row source table, we use the
    cache (``tag_features_standardize_first.sqlite``) as a
    pre-computed list of the standardized (key, value) pairs that
    survived the ``count_all >= 500`` filter. We ATTACH the cache
    and use it as a filter inside the GROUP BY. This is much
    cheaper than a full 192 M-row LOWER+GROUP BY scan.

    If the cache is not present, we fall back to the full scan
    (~100-200 s on the project's external drive).
    """
    # Default cache path: alongside the source DB on the Seagate drive.
    if cache_path is None:
        cache_path = "/Volumes/Seagate M3/tag_features_standardize_first.sqlite"
    cache_exists = (
        cache_path is not None
        and isinstance(cache_path, (str, Path))
        and str(cache_path) != ""
        and Path(cache_path).exists()
    )

    if cache_exists:
        # Attach the cache as a side DB, then use it as a filter.
        # The cache has 225,684 rows of lowercased (key, value).
        # The JOIN uses LOWER on the source side; SQLite does a
        # hash join which is fast.
        db.execute(f"ATTACH DATABASE ? AS cache", (str(cache_path),))
        sql = """
            SELECT
                LOWER(t.key)   AS key,
                LOWER(t.value) AS value,
                SUM(t.count_nodes)     AS count_nodes,
                SUM(t.count_ways)      AS count_ways,
                SUM(t.count_relations) AS count_relations
            FROM main.tags AS t
            JOIN cache.tag_features AS c
              ON c.key = LOWER(t.key) AND c.value = LOWER(t.value)
            GROUP BY LOWER(t.key), LOWER(t.value)
        """
        try:
            return pd.read_sql_query(sql, db)
        finally:
            db.execute("DETACH DATABASE cache")
    else:
        # Fallback: full table scan. Slow but correct.
        sql = """
            SELECT
                LOWER(key)   AS key,
                LOWER(value) AS value,
                SUM(count_nodes)     AS count_nodes,
                SUM(count_ways)      AS count_ways,
                SUM(count_relations) AS count_relations
            FROM tags
            GROUP BY LOWER(key), LOWER(value)
        """
        return pd.read_sql_query(sql, db)


def supercluster_element_type_stats(
    base_keys: Iterable[str],
    cluster_medoids: Union[str, Path, pd.DataFrame],
    cluster_memberships: Union[str, Path, pd.DataFrame],
    source_db: Union[str, Path, sqlite3.Connection],
    source_cache: Union[str, Path, None] = None,
) -> pd.DataFrame:
    """Compute the per-supercluster element-type split.

    Parameters
    ----------
    base_keys
        The supercluster base keys to look up. May be in any
        order; the returned DataFrame preserves the input order.
        Empty input returns an empty DataFrame with the canonical
        column schema.
    cluster_medoids
        Path to (or DataFrame of) the medoid CSV. Must have
        ``cluster_id`` and ``medoid_feature`` columns.
    cluster_memberships
        Path to (or DataFrame of) the cluster_memberships CSV.
        Must have ``cluster_id``, ``key``, ``value``, ``feature``,
        ``count_all`` columns.
    source_db
        Path to (or sqlite3.Connection to) the source taginfo DB
        that has a ``tags(key, value, count_all, count_nodes,
        count_ways, count_relations)`` table.

    Returns
    -------
    pandas.DataFrame
        One row per supercluster base key, with the columns listed
        in :data:`SUPERCLUSTER_STATS_COLUMNS`. Base keys with no
        clusters get a zero row (not dropped).
    """
    base_keys_list = [str(b).strip() for b in base_keys]
    if not base_keys_list:
        return pd.DataFrame(columns=SUPERCLUSTER_STATS_COLUMNS)

    # 1. Map cluster_id -> supercluster base_key, from the medoid file.
    #    Skip the noise row (cluster_id = -1).
    medoids = _load_medoids(cluster_medoids)
    if not medoids.empty and "cluster_id" in medoids.columns:
        real_medoids = medoids[medoids["cluster_id"] != -1].copy()
        if not real_medoids.empty:
            real_medoids["supercluster_bk"] = (
                real_medoids["medoid_feature"].map(parse_base_key)
            )
            cluster_to_bk = dict(
                zip(real_medoids["cluster_id"].astype("int64"),
                    real_medoids["supercluster_bk"])
            )
        else:
            cluster_to_bk = {}
    else:
        cluster_to_bk = {}

    # 2. Read cluster memberships, drop noise, attach the supercluster
    #    base key of each member's cluster. Members whose cluster is
    #    in the noise are dropped; members of a real cluster that
    #    happens to have a different own base_key are kept (they
    #    belong to the supercluster via the cluster's medoid).
    #    Then we filter to only the requested base keys so the output
    #    has one row per input base key (plus zero rows for any input
    #    base key that has no clusters).
    requested = set(base_keys_list)
    mem = _load_members(cluster_memberships)
    if not mem.empty and "cluster_id" in mem.columns:
        real_mem = mem[mem["cluster_id"] != -1].copy()
        if not real_mem.empty:
            real_mem["cluster_id"] = real_mem["cluster_id"].astype("int64")
            real_mem["supercluster_bk"] = real_mem["cluster_id"].map(cluster_to_bk)
            real_mem = real_mem.dropna(subset=["supercluster_bk"])
            real_mem["supercluster_bk"] = real_mem["supercluster_bk"].astype(str)
            # Filter to only the requested base keys.
            real_mem = real_mem[real_mem["supercluster_bk"].isin(requested)]
        else:
            real_mem = pd.DataFrame()
    else:
        real_mem = pd.DataFrame()

    # 3. Pre-aggregate the source DB by LOWER(key), LOWER(value) to
    #    fold case/whitespace variants together. This is the slow
    #    step (~100 s on the real 14 GB source, but with the cache
    #    filter it is much faster).
    con = _resolve_source_db(source_db)
    try:
        src = _aggregate_source_db_by_lower(con, cache_path=source_cache)
    finally:
        # We do not close con if the caller passed it in.
        if not isinstance(source_db, sqlite3.Connection):
            con.close()

    # 4. Join cluster members to the source DB on (key, value). The
    #    cluster_memberships are already lowercased, so a plain join
    #    finds the right element-type split (case variants were
    #    folded in step 3). Missing matches become 0 below. When
    #    real_mem is empty (no real clusters), skip the join.
    if real_mem.empty:
        joined = pd.DataFrame()
    else:
        joined = real_mem.merge(
            src,
            on=["key", "value"],
            how="left",
        )
        for col in ("count_nodes", "count_ways", "count_relations"):
            joined[col] = joined[col].fillna(0).astype("int64")

    # 5. Group by supercluster base key and sum.
    if joined.empty:
        grouped = pd.DataFrame(columns=[
            "base_key", "n_clusters", "n_tags", "count_all",
            "count_nodes", "count_ways", "count_relations",
        ])
    else:
        grouped = (
            joined.groupby("supercluster_bk", as_index=False)
            .agg(
                n_clusters=("cluster_id", "nunique"),
                n_tags=("feature", "count"),
                count_all=("count_all", "sum"),
                count_nodes=("count_nodes", "sum"),
                count_ways=("count_ways", "sum"),
                count_relations=("count_relations", "sum"),
            )
            .rename(columns={"supercluster_bk": "base_key"})
        )

    # 6. Percentages and is_polygon_friendly. count_all is the sum of
    #    cluster_memberships.count_all, so it is always > 0 when
    #    grouped is non-empty; the safe_total is here for the
    #    empty-row case below.
    for col in ("count_all", "count_nodes", "count_ways", "count_relations"):
        if col in grouped.columns:
            grouped[col] = grouped[col].astype("int64")

    if grouped.empty:
        grouped = grouped.assign(
            pct_nodes=pd.Series(dtype="float64"),
            pct_ways=pd.Series(dtype="float64"),
            pct_relations=pd.Series(dtype="float64"),
            is_polygon_friendly=pd.Series(dtype="bool"),
        )
    else:
        safe_total = grouped["count_all"].where(
            grouped["count_all"] > 0, other=1
        )
        grouped["pct_nodes"]     = (grouped["count_nodes"]     / safe_total * 100).round(2)
        grouped["pct_ways"]      = (grouped["count_ways"]      / safe_total * 100).round(2)
        grouped["pct_relations"] = (grouped["count_relations"] / safe_total * 100).round(2)
        grouped["is_polygon_friendly"] = (
            (grouped["count_ways"] + grouped["count_relations"]) / safe_total
        ) >= POLYGON_FRIENDLY_THRESHOLD

    # 7. Add zero rows for base keys that are not represented in the
    #    medoid file at all (the user may have labeled them but the
    #    pipeline never produced a cluster with that medoid).
    found = set(grouped["base_key"].tolist()) if not grouped.empty else set()
    missing = [b for b in base_keys_list if b not in found]
    if missing:
        zeros = pd.DataFrame({
            "base_key":            missing,
            "n_clusters":          [0] * len(missing),
            "n_tags":              [0] * len(missing),
            "count_all":           [0] * len(missing),
            "count_nodes":         [0] * len(missing),
            "count_ways":          [0] * len(missing),
            "count_relations":     [0] * len(missing),
            "pct_nodes":           [0.0] * len(missing),
            "pct_ways":            [0.0] * len(missing),
            "pct_relations":       [0.0] * len(missing),
            "is_polygon_friendly": [False] * len(missing),
        })
        grouped = pd.concat([grouped, zeros], ignore_index=True)

    # 8. Reorder columns and rows. Preserve caller's input order.
    order_index = {b: i for i, b in enumerate(base_keys_list)}
    grouped["_order"] = grouped["base_key"].map(order_index)
    grouped = (
        grouped.sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )
    grouped = grouped[SUPERCLUSTER_STATS_COLUMNS]

    return grouped
