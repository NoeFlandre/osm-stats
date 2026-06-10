"""Group cluster medoids by their base OSM key.

After HDBSCAN produces ~9,000 clusters and the medoid step assigns each one
a representative tag, the user wants a single table that says: "across all
clusters, how dominant is the ``landuse`` family? the ``addr`` family?".
This module groups medoids by their base key (parsed from the medoid
string) and aggregates three things:

* ``cluster_count`` - how many clusters landed in this family
* ``total_count_all`` - sum of occurrences across all clusters in the family
* ``representative_medoids`` - the top-N medoids by count, joined with ``"; "``

The result is sorted by ``total_count_all`` descending so the most
voluminous key families come first.
"""
from __future__ import annotations

import pandas as pd

from src.core.features.base_key import parse_base_key


def profile_clusters_by_base_key(
    medoids_df: pd.DataFrame, top_n: int = 5
) -> pd.DataFrame:
    """Group *medoids_df* by base key and aggregate the cluster stats.

    The noise row (``cluster_id == -1``) is excluded. The returned frame
    is sorted by ``total_count_all`` descending.
    """
    if medoids_df.empty:
        return pd.DataFrame(
            columns=[
                "base_key",
                "cluster_count",
                "total_count_all",
                "representative_medoids",
            ]
        )

    real = medoids_df[medoids_df["cluster_id"] != -1].copy()
    real["base_key"] = real["medoid_feature"].map(parse_base_key)

    rows = []
    for base_key, group in real.groupby("base_key"):
        top = (
            group.sort_values("total_count_all", ascending=False)
            .head(top_n)["medoid_feature"]
            .tolist()
        )
        rows.append(
            {
                "base_key": base_key,
                "cluster_count": int(len(group)),
                "total_count_all": int(group["total_count_all"].sum()),
                "representative_medoids": "; ".join(top),
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values("total_count_all", ascending=False).reset_index(drop=True)
    return out
