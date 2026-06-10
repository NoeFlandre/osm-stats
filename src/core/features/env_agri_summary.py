"""Aggregate stats over the env/agri per-cluster breakdown.

The numbers come straight from the persisted per-cluster medoid file
(``output/cluster_medoids.csv``) filtered to the env/agri whitelist.
Two semantics are exposed:

* ``total_occurrences`` - the sum of per-cluster ``total_count_all``
  across the whitelisted clusters. This is the env/agri slice of the
  OSM tag volume that survived HDBSCAN clustering.
* ``top_base_key`` and ``top_base_key_occurrences`` - the family
  (e.g. ``landuse``, ``species``) with the largest combined volume.
"""
from __future__ import annotations

import pandas as pd

from src.core.features.breakdown import env_agri_breakdown_df
from src.core.features.env_agri_whitelist import ENVI_AGRI_BASE_KEYS


def env_agri_summary() -> dict:
    """Return aggregate stats for the env/agri per-cluster breakdown."""
    df: pd.DataFrame = env_agri_breakdown_df()

    if df.empty:
        return {
            "n_base_keys": 0,
            "n_clusters": 0,
            "total_occurrences": 0,
            "top_base_key": "",
            "top_base_key_occurrences": 0,
        }

    by_key = df.groupby("base_key")["total_count_all"].sum().sort_values(ascending=False)
    top_key = by_key.index[0]
    top_occ = int(by_key.iloc[0])

    return {
        "n_base_keys": len(ENVI_AGRI_BASE_KEYS),
        "n_clusters": int(len(df)),
        "total_occurrences": int(df["total_count_all"].sum()),
        "top_base_key": str(top_key),
        "top_base_key_occurrences": top_occ,
    }


def render_env_agri_summary_markdown() -> str:
    s = env_agri_summary()
    lines = [
        "| stat | value |",
        "| --- | ---: |",
        f"| number of env/agri base keys | {s['n_base_keys']} |",
        f"| number of env/agri clusters | {s['n_clusters']} |",
        f"| total occurrences (env/agri) | {s['total_occurrences']:,} |",
        f"| top base key | {s['top_base_key']} |",
        f"| top base key occurrences | {s['top_base_key_occurrences']:,} |",
    ]
    return "\n".join(lines)
