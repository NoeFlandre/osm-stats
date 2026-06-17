"""Tests for ``src.core.db.supercluster_element_type_stats``.

The pipeline produces 8,832 real clusters from HDBSCAN. Each cluster
gets a medoid (the actual member closest to the centroid), and the
medoid's base key determines which supercluster the cluster belongs
to. A supercluster = the union of all members of all clusters whose
medoid's base key matches.

This module computes the element-type split of the supercluster's
actual contents (not of every (key, value) pair globally with that
base key in the source DB). The two views differ because:

- a supercluster can contain members with different base keys than
  the supercluster's own (e.g. a cluster with medoid
  ``tree|species:oak`` may also contain ``forest|species:oak``);
- the source DB view includes source rows that ended up in noise
  (cluster_id = -1), which are not in any supercluster;
- the source DB view includes case/whitespace variants of the same
  standardized tag, which the cluster view folds together.

We join cluster_memberships to the source DB to get the
``count_nodes / count_ways / count_relations`` per member. To handle
case variants, the source DB is pre-aggregated by
``LOWER(key), LOWER(value)`` so that ``landuse`` and ``Landuse``
collapse to the same element-type split.

Noise (cluster_id = -1) is excluded by the user request.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from src.core.db.element_type_stats import POLYGON_FRIENDLY_THRESHOLD
from src.core.db.supercluster_element_type_stats import (
    SUPERCLUSTER_STATS_COLUMNS,
    supercluster_element_type_stats,
)


def _make_source_db(rows: list[tuple]) -> sqlite3.Connection:
    """In-memory SQLite with a ``tags`` table populated from *rows*.

    Each row is ``(key, value, count_all, count_nodes, count_ways,
    count_relations)``. Mixed case is allowed (the function should
    LOWER() it before joining).
    """
    con = sqlite3.connect(":memory:")
    con.execute(
        """
        CREATE TABLE tags (
            key              VARCHAR,
            value            VARCHAR,
            count_all        INTEGER,
            count_nodes      INTEGER,
            count_ways       INTEGER,
            count_relations  INTEGER
        );
        """
    )
    con.executemany("INSERT INTO tags VALUES (?, ?, ?, ?, ?, ?)", rows)
    con.commit()
    return con


#: All tests pass a non-existent cache path so the function falls
#: back to the full table scan (no auto-detection of the real
#: ``/Volumes/Seagate M3/`` cache, which would not match the
#: in-memory test data).
NO_CACHE = "/tmp/_test_does_not_exist_cache.sqlite"


def _make_medoids(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal medoid DataFrame.

    Each row needs ``cluster_id`` and ``medoid_feature``. The
    function only reads those two columns plus optionally
    ``cluster_size`` and ``total_count_all`` (which it does not use,
    so they are optional).
    """
    return pd.DataFrame(rows)


def _make_members(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal cluster_memberships DataFrame.

    Each row needs ``cluster_id``, ``key``, ``value``, ``feature``,
    ``count_all``.
    """
    return pd.DataFrame(rows)


# ---------- core behavior ----------

def test_polygon_heavy_supercluster_is_flagged_polygon_friendly():
    """A supercluster whose members are mostly on ways+relations is
    polygon-friendly."""
    src = _make_source_db([("landuse", "farmland", 100, 0, 95, 5)])
    med = _make_medoids([{
        "cluster_id": 0, "medoid_feature": "landuse|farmland",
        "cluster_size": 1, "total_count_all": 100,
    }])
    mem = _make_members([{
        "cluster_id": 0, "base_key": "landuse",
        "key": "landuse", "value": "farmland",
        "feature": "landuse|farmland", "count_all": 100,
    }])
    out = supercluster_element_type_stats(["landuse"], med, mem, src, source_cache=NO_CACHE)
    row = out.iloc[0]
    assert row["base_key"] == "landuse"
    assert row["n_clusters"] == 1
    assert row["n_tags"] == 1
    assert row["count_all"] == 100
    assert row["count_nodes"] == 0
    assert row["count_ways"] == 95
    assert row["count_relations"] == 5
    assert bool(row["is_polygon_friendly"]) is True


def test_point_heavy_supercluster_is_not_polygon_friendly():
    """A supercluster whose members are mostly on nodes is not
    polygon-friendly."""
    src = _make_source_db([("monitoring:water_quality", "yes", 200, 195, 5, 0)])
    med = _make_medoids([{
        "cluster_id": 0, "medoid_feature": "monitoring:water_quality|yes",
        "cluster_size": 1, "total_count_all": 200,
    }])
    mem = _make_members([{
        "cluster_id": 0, "base_key": "monitoring",
        "key": "monitoring:water_quality", "value": "yes",
        "feature": "monitoring:water_quality|yes", "count_all": 200,
    }])
    out = supercluster_element_type_stats(["monitoring"], med, mem, src, source_cache=NO_CACHE)
    row = out.iloc[0]
    assert bool(row["is_polygon_friendly"]) is False
    assert row["pct_nodes"] == pytest.approx(97.5)


def test_noise_is_excluded():
    """A noise row in cluster_memberships is NOT counted in the
    supercluster stats, even if its own base_key matches."""
    src = _make_source_db([
        ("landuse", "farmland", 100, 0, 95, 5),       # real member
        ("landuse", "lonely_orphan", 50, 50, 0, 0),   # noise row
    ])
    med = _make_medoids([{
        "cluster_id": 0, "medoid_feature": "landuse|farmland",
        "cluster_size": 1, "total_count_all": 100,
    }])
    mem = _make_members([
        {"cluster_id": 0, "base_key": "landuse",
         "key": "landuse", "value": "farmland",
         "feature": "landuse|farmland", "count_all": 100},
        {"cluster_id": -1, "base_key": "landuse",  # noise
         "key": "landuse", "value": "lonely_orphan",
         "feature": "landuse|lonely_orphan", "count_all": 50},
    ])
    out = supercluster_element_type_stats(["landuse"], med, mem, src, source_cache=NO_CACHE)
    row = out.iloc[0]
    assert row["n_tags"] == 1                # noise excluded
    assert row["count_all"] == 100            # only real-cluster occ
    assert row["count_nodes"] == 0
    assert row["count_ways"] == 95


def test_cross_base_key_members_count():
    """A supercluster contains all cluster members, even if some
    members' own base key differs from the supercluster's base key.

    Example: cluster with medoid ``tree|species:oak`` (base_key=tree)
    contains a member ``forest|species:oak`` (own base_key=forest).
    That member counts for the ``tree`` supercluster, because the
    cluster was assigned to it via its medoid.
    """
    src = _make_source_db([
        ("tree", "species:oak", 100, 0, 95, 5),
        ("forest", "species:oak", 50, 0, 48, 2),  # different base_key
    ])
    med = _make_medoids([{
        "cluster_id": 0, "medoid_feature": "tree|species:oak",
        "cluster_size": 2, "total_count_all": 150,
    }])
    mem = _make_members([
        {"cluster_id": 0, "base_key": "tree",
         "key": "tree", "value": "species:oak",
         "feature": "tree|species:oak", "count_all": 100},
        {"cluster_id": 0, "base_key": "forest",  # own base_key = forest
         "key": "forest", "value": "species:oak",
         "feature": "forest|species:oak", "count_all": 50},
    ])
    out = supercluster_element_type_stats(["tree"], med, mem, src, source_cache=NO_CACHE)
    row = out.iloc[0]
    # Both members are in the "tree" supercluster because the cluster
    # was assigned there via its medoid.
    assert row["n_tags"] == 2
    assert row["count_all"] == 150
    assert row["count_ways"] == 95 + 48  # both contribute
    assert row["count_nodes"] == 0


def test_case_variants_in_source_db_are_aggregated():
    """If the source DB has both ``landuse=farmland`` and
    ``Landuse=farmland``, both contribute to the element-type split
    of the standardized ``landuse|farmland`` member (the cluster's
    count_all is the sum of both)."""
    src = _make_source_db([
        ("landuse", "farmland", 60, 0, 58, 2),
        ("Landuse", "Farmland", 40, 0, 38, 2),
    ])
    med = _make_medoids([{
        "cluster_id": 0, "medoid_feature": "landuse|farmland",
        "cluster_size": 1, "total_count_all": 100,
    }])
    mem = _make_members([{
        "cluster_id": 0, "base_key": "landuse",
        "key": "landuse", "value": "farmland",
        "feature": "landuse|farmland", "count_all": 100,
    }])
    out = supercluster_element_type_stats(["landuse"], med, mem, src, source_cache=NO_CACHE)
    row = out.iloc[0]
    assert row["count_all"] == 100  # cluster's count_all
    assert row["count_ways"] == 58 + 38  # both variants contribute
    assert row["count_relations"] == 4


def test_supercluster_with_multiple_clusters_sums_them():
    """A supercluster can contain multiple clusters. The stats sum
    over all of them."""
    src = _make_source_db([
        ("landuse", "farmland", 100, 0, 95, 5),
        ("landuse", "orchard",  50, 0, 48, 2),
    ])
    med = _make_medoids([
        {"cluster_id": 0, "medoid_feature": "landuse|farmland",
         "cluster_size": 1, "total_count_all": 100},
        {"cluster_id": 1, "medoid_feature": "landuse|orchard",
         "cluster_size": 1, "total_count_all": 50},
    ])
    mem = _make_members([
        {"cluster_id": 0, "base_key": "landuse",
         "key": "landuse", "value": "farmland",
         "feature": "landuse|farmland", "count_all": 100},
        {"cluster_id": 1, "base_key": "landuse",
         "key": "landuse", "value": "orchard",
         "feature": "landuse|orchard", "count_all": 50},
    ])
    out = supercluster_element_type_stats(["landuse"], med, mem, src, source_cache=NO_CACHE)
    row = out.iloc[0]
    assert row["n_clusters"] == 2
    assert row["n_tags"] == 2
    assert row["count_all"] == 150
    assert row["count_ways"] == 143
    assert row["count_relations"] == 7


def test_base_key_with_zero_clusters_returns_zero_row_not_dropped():
    """A base key that has no clusters in the medoid file (so its
    supercluster is empty) appears in the output with zeros; it is
    NOT dropped."""
    src = _make_source_db([("landuse", "farmland", 100, 0, 95, 5)])
    med = _make_medoids([{
        "cluster_id": 0, "medoid_feature": "natural|tree",
        "cluster_size": 1, "total_count_all": 100,
    }])  # only a "natural" cluster; no "landuse" cluster
    mem = _make_members([{
        "cluster_id": 0, "base_key": "natural",
        "key": "natural", "value": "tree",
        "feature": "natural|tree", "count_all": 100,
    }])
    out = supercluster_element_type_stats(["landuse", "natural"], med, mem, src, source_cache=NO_CACHE)
    assert list(out["base_key"]) == ["landuse", "natural"]
    landuse = out[out.base_key == "landuse"].iloc[0]
    assert landuse["n_clusters"] == 0
    assert landuse["n_tags"] == 0
    assert landuse["count_all"] == 0
    assert bool(landuse["is_polygon_friendly"]) is False
    natural = out[out.base_key == "natural"].iloc[0]
    assert natural["n_clusters"] == 1
    assert natural["count_all"] == 100


def test_unknown_base_key_returns_zero_row():
    """A base key not represented in the medoid file at all (e.g.
    the user labeled it but the pipeline never produced a cluster
    with that medoid) gets a zero row."""
    src = _make_source_db([("landuse", "farmland", 100, 0, 95, 5)])
    med = _make_medoids([])  # empty medoid file
    mem = _make_members([])   # empty memberships
    out = supercluster_element_type_stats(["nowhere_base_key"], med, mem, src, source_cache=NO_CACHE)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["base_key"] == "nowhere_base_key"
    assert row["count_all"] == 0
    assert bool(row["is_polygon_friendly"]) is False


def test_empty_input_returns_empty_dataframe_with_canonical_columns():
    """An empty base_keys list returns an empty DataFrame with the
    canonical schema."""
    src = _make_source_db([])
    med = _make_medoids([])
    mem = _make_members([])
    out = supercluster_element_type_stats([], med, mem, src, source_cache=NO_CACHE)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == SUPERCLUSTER_STATS_COLUMNS
    assert len(out) == 0


def test_output_preserves_input_order():
    """The output preserves the order of the input base_keys list,
    not the order in the medoid file."""
    src = _make_source_db([
        ("landuse", "farmland", 100, 0, 95, 5),
        ("natural", "tree", 50, 0, 48, 2),
        ("waterway", "stream", 30, 0, 30, 0),
    ])
    med = _make_medoids([
        {"cluster_id": 0, "medoid_feature": "landuse|farmland"},
        {"cluster_id": 1, "medoid_feature": "natural|tree"},
        {"cluster_id": 2, "medoid_feature": "waterway|stream"},
    ])
    mem = _make_members([
        {"cluster_id": 0, "base_key": "landuse",
         "key": "landuse", "value": "farmland",
         "feature": "landuse|farmland", "count_all": 100},
        {"cluster_id": 1, "base_key": "natural",
         "key": "natural", "value": "tree",
         "feature": "natural|tree", "count_all": 50},
        {"cluster_id": 2, "base_key": "waterway",
         "key": "waterway", "value": "stream",
         "feature": "waterway|stream", "count_all": 30},
    ])
    out = supercluster_element_type_stats(
        ["waterway", "landuse", "natural"], med, mem, src,
    )
    assert list(out["base_key"]) == ["waterway", "landuse", "natural"]


def test_member_not_found_in_source_db_contributes_zero():
    """A cluster member whose (key, value) is not in the source DB
    contributes 0 to the element-type split (rather than NaN).
    """
    src = _make_source_db([])  # empty source DB
    med = _make_medoids([{
        "cluster_id": 0, "medoid_feature": "landuse|farmland",
    }])
    mem = _make_members([{
        "cluster_id": 0, "base_key": "landuse",
        "key": "landuse", "value": "farmland",
        "feature": "landuse|farmland", "count_all": 100,
    }])
    out = supercluster_element_type_stats(["landuse"], med, mem, src, source_cache=NO_CACHE)
    row = out.iloc[0]
    assert row["count_all"] == 100       # from memberships
    assert row["count_nodes"] == 0        # not in source DB
    assert row["count_ways"] == 0
    assert row["count_relations"] == 0


def test_path_or_connection_for_source_db():
    """The source_db argument accepts either a Connection or a Path,
    matching the element_type_stats convention."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "tags.sqlite"
        con = sqlite3.connect(db)
        con.execute(
            "CREATE TABLE tags (key VARCHAR, value VARCHAR, "
            "count_all INTEGER, count_nodes INTEGER, "
            "count_ways INTEGER, count_relations INTEGER)"
        )
        con.executemany(
            "INSERT INTO tags VALUES (?, ?, ?, ?, ?, ?)",
            [("landuse", "farmland", 100, 0, 95, 5)],
        )
        con.commit()
        con.close()

        med = pd.DataFrame([{
            "cluster_id": 0, "medoid_feature": "landuse|farmland",
        }])
        mem = pd.DataFrame([{
            "cluster_id": 0, "base_key": "landuse",
            "key": "landuse", "value": "farmland",
            "feature": "landuse|farmland", "count_all": 100,
        }])

        out = supercluster_element_type_stats(["landuse"], med, mem, db)
        assert len(out) == 1
        assert out.iloc[0]["count_all"] == 100


def test_canonical_columns_constant():
    """SUPERCLUSTER_STATS_COLUMNS is the canonical schema for the
    output DataFrame."""
    assert isinstance(SUPERCLUSTER_STATS_COLUMNS, list)
    for col in (
        "base_key", "n_clusters", "n_tags", "count_all",
        "count_nodes", "count_ways", "count_relations",
        "pct_nodes", "pct_ways", "pct_relations",
        "is_polygon_friendly",
    ):
        assert col in SUPERCLUSTER_STATS_COLUMNS
