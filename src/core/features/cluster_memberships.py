"""Persist the per-tag cluster membership mapping for a clustering run.

The medoid file (``cluster_medoids*.csv``) gives one row per cluster
(plus one summary row for the noise bucket) with the cluster's medoid,
member count, and total occurrences. That view is a *summary*: it
says "cluster 0 has 8 members and 121k total occurrences", but it does
not say which tags those 8 members are.

This module writes the complementary *membership* view: one row per
tag in the cache, with the cluster it landed in. With both files in
hand, the user can decide which clusters to keep by inspecting the
memberships:

* the medoid file answers "what clusters exist, and how big are they?"
* the membership file answers "what tags live in each cluster?"

The membership file is the **raw, unfiltered** output of HDBSCAN.
The env/agri filter is applied later by the maintainer or by an
external LLM step; this module never filters.

The output schema is::

    cluster_id, base_key, key, value, feature, count_all

``cluster_id`` is the integer HDBSCAN label; ``-1`` is the noise
bucket. ``base_key`` is the OSM key namespace root derived from the
key (everything before the first colon). ``feature`` is the
standardized ``"<key>|<value>"`` string. The original ``count_all``
is preserved so the user can compute per-cluster totals themselves.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from src.core.features.base_key import parse_base_key


MEMBERSHIPS_COLUMNS: list[str] = [
    "cluster_id",
    "base_key",
    "key",
    "value",
    "feature",
    "count_all",
]


def save_cluster_memberships(
    cluster_ids: np.ndarray,
    df: pd.DataFrame,
    output_path: Union[str, Path],
) -> Path:
    """Write a per-tag cluster membership CSV.

    The output has columns ``cluster_id, base_key, key, value,
    feature, count_all`` (see module docstring). The input
    *cluster_ids* is the HDBSCAN output (an integer array, ``-1`` for
    noise). The input *df* must have columns ``key``, ``value``,
    ``feature``, and ``count_all`` in any order; rows must align with
    *cluster_ids* row-for-row.

    Returns the path that was written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cluster_ids = np.asarray(cluster_ids)
    if len(cluster_ids) != len(df):
        raise ValueError(
            f"cluster_ids length ({len(cluster_ids)}) does not match "
            f"df length ({len(df)})"
        )

    out = df.copy()
    out.insert(0, "cluster_id", cluster_ids.astype(np.int64))
    # Derive the base key from the key column. If the key column is
    # missing the standard "key|value" pipe, fall back to deriving it
    # from the feature column so the helper still works on a cache
    # that hasn't been re-joined with its original key/value.
    out["base_key"] = out["feature"].map(parse_base_key)
    out = out[MEMBERSHIPS_COLUMNS]
    out.to_csv(output_path, index=False)
    return output_path
