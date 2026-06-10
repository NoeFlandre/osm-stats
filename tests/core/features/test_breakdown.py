"""The env/agri breakdown must produce one row per real cluster for
the whitelisted base keys, sourced from a persisted per-cluster
medoid file (``output/cluster_medoids.parquet``) that carries the
true per-cluster occurrence total.

The breakdown is built bottom-up from the cluster medoid file (not
the cluster profile Markdown), so each row reflects one real cluster
with its real ``cluster_id``, ``cluster_size`` (number of members),
and ``total_count_all`` (sum of occurrences across those members).
"""
from pathlib import Path

import pandas as pd

from src.core.features.env_agri_whitelist import ENVI_AGRI_BASE_KEYS
from src.core.features.breakdown import (
    env_agri_breakdown_df,
    render_breakdown_markdown,
)


MEDOIDS_PATH = Path("output/cluster_medoids.csv")


# --- medoid file existence ---------------------------------------------


def test_medoids_parquet_exists():
    """The pipeline must persist per-cluster medoids so the breakdown
    can read real per-cluster counts without re-running HDBSCAN."""
    assert MEDOIDS_PATH.exists(), (
        f"missing {MEDOIDS_PATH}: re-run scripts/profile_clusters.py to "
        f"produce the per-cluster medoid file"
    )


# --- env_agri_breakdown_df ---------------------------------------------


def test_breakdown_returns_dataframe():
    df = env_agri_breakdown_df()
    assert not df.empty
    expected_cols = ["base_key", "cluster_id", "medoid", "cluster_size", "total_count_all"]
    assert list(df.columns) == expected_cols


def test_breakdown_only_contains_whitelisted_base_keys():
    df = env_agri_breakdown_df()
    assert set(df["base_key"]) <= ENVI_AGRI_BASE_KEYS


def test_breakdown_cluster_ids_are_unique():
    df = env_agri_breakdown_df()
    assert df["cluster_id"].is_unique


def test_breakdown_cluster_size_is_integer():
    df = env_agri_breakdown_df()
    # ``cluster_size`` is the number of OSM tag features that landed
    # in the cluster. It must be a positive integer.
    assert pd.api.types.is_integer_dtype(df["cluster_size"])
    assert (df["cluster_size"] > 0).all()


def test_breakdown_total_count_all_is_integer():
    df = env_agri_breakdown_df()
    # ``total_count_all`` is the sum of ``count_all`` across the
    # cluster's members. It must be a positive integer.
    assert pd.api.types.is_integer_dtype(df["total_count_all"])
    assert (df["total_count_all"] > 0).all()


def test_breakdown_per_cluster_count_matches_medoid_file():
    """Each row's total_count_all must equal the same field in the
    persisted medoid file. The breakdown is a filter on that file,
    not a recomputation."""
    df = env_agri_breakdown_df()
    src = pd.read_csv(MEDOIDS_PATH)
    src_by_id = src.set_index("cluster_id")
    for _, row in df.iterrows():
        cid = int(row["cluster_id"])
        assert int(row["total_count_all"]) == int(src_by_id.loc[cid, "total_count_all"])


def test_breakdown_medoid_matches_medoid_file():
    df = env_agri_breakdown_df()
    src = pd.read_csv(MEDOIDS_PATH)
    src_by_id = src.set_index("cluster_id")
    for _, row in df.iterrows():
        cid = int(row["cluster_id"])
        assert row["medoid"] == src_by_id.loc[cid, "medoid_feature"]


# --- render_breakdown_markdown -----------------------------------------


def test_render_returns_string():
    md = render_breakdown_markdown(env_agri_breakdown_df())
    assert isinstance(md, str)


def test_render_has_header_and_separator():
    md = render_breakdown_markdown(env_agri_breakdown_df())
    assert "| base_key |" in md
    assert "| --- |" in md


def test_render_includes_one_row_per_cluster():
    df = env_agri_breakdown_df()
    md = render_breakdown_markdown(df)
    body_rows = [
        ln
        for ln in md.splitlines()
        if ln.startswith("| ") and "base_key" not in ln and "---" not in ln
    ]
    assert len(body_rows) == len(df)


def test_render_includes_medoid_for_every_cluster():
    df = env_agri_breakdown_df()
    md = render_breakdown_markdown(df)
    for medoid in df["medoid"]:
        if not medoid:
            continue
        # The medoid in the rendered table has its '|' escaped.
        assert medoid.replace("|", "\\|") in md


def test_render_pipe_in_medoid_escaped():
    df = env_agri_breakdown_df()
    md = render_breakdown_markdown(df)
    body_rows = [
        ln
        for ln in md.splitlines()
        if ln.startswith("| ") and "base_key" not in ln and "---" not in ln
    ]
    assert body_rows, "expected at least one body row"
    for line in body_rows:
        # 5 columns + leading '' + trailing '' = 7 cells when split on
        # un-escaped pipes.
        cleaned = line.replace("\\|", "\x00")
        cells = [c.strip() for c in cleaned.split("|")]
        assert len(cells) == 7, f"row not 5 columns: {line!r}"
