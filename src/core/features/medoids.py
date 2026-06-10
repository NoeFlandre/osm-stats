"""Compute a medoid for every cluster, plus a summary row for the noise bucket.

A *medoid* is the actual data point inside a cluster that is closest to the
cluster's centroid. We use it to label each cluster with a real tag string
(e.g. ``addr:street|hauptstraße``) instead of an abstract vector. This
makes the cluster summary human-readable and gives the profiling stage a
representative feature to extract the base OSM key from.

The noise bucket (label ``-1``) is summarized as a single row. Individual
high-volume noise tags are not micro-clustered - they appear in the
"noise" row's totals and can be inspected separately if needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_cluster_medoids(
    features: list,
    dense: np.ndarray,
    labels: np.ndarray,
    counts: list,
) -> pd.DataFrame:
    """Return one row per cluster (and one for the noise bucket).

    Each row has: ``cluster_id``, ``medoid_feature``, ``cluster_size``,
    ``total_count_all``. Non-noise clusters are ordered by cluster_id
    ascending; the noise row, if present, comes last.
    """
    if not (len(features) == len(dense) == len(labels) == len(counts)):
        raise ValueError(
            "compute_cluster_medoids: features, dense, labels, counts must "
            "have the same length"
        )
    dense = np.asarray(dense)
    labels = np.asarray(labels, dtype=np.int64)

    rows = []
    unique_labels = sorted(set(int(c) for c in labels))
    for cid in unique_labels:
        mask = labels == cid
        members = dense[mask]
        member_features = [f for f, m in zip(features, mask) if m]
        member_counts = [c for c, m in zip(counts, mask) if m]
        size = int(mask.sum())

        if cid == -1:
            # The "noise" bucket: one summary row. The top noise features
            # by count are exposed in a separate column so the profiler
            # can summarize them.
            top = sorted(
                zip(member_features, member_counts),
                key=lambda x: -x[1],
            )[:3]
            rows.append(
                {
                    "cluster_id": -1,
                    "medoid_feature": top[0][0] if top else "",
                    "cluster_size": size,
                    "total_count_all": int(sum(member_counts)),
                    "top_features": ";".join(f for f, _ in top),
                }
            )
            continue

        # Real cluster: pick the member closest to the centroid.
        centroid = members.mean(axis=0)
        distances = np.linalg.norm(members - centroid, axis=1)
        medoid_idx = int(np.argmin(distances))
        rows.append(
            {
                "cluster_id": cid,
                "medoid_feature": member_features[medoid_idx],
                "cluster_size": size,
                "total_count_all": int(sum(member_counts)),
            }
        )

    return pd.DataFrame(rows)
