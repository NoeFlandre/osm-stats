"""HDBSCAN clustering on the dense SVD-reduced feature space.

The pipeline so far: 224,123 tag strings -> TF-IDF (sparse, 224k x ~400k)
-> TruncatedSVD (dense, 224k x 50). The SVD output is the right shape
for HDBSCAN: dense, low-dimensional, euclidean-safe.

HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications
with Noise) groups points that are close *and* dense together, and pushes
isolated points into a "noise" bucket labelled -1. This is exactly what
we want for OSM tags: a handful of large, dense clusters for the popular
land-use and natural keys, and a long tail of one-off tags (typos, custom
notations, very rare values) that the user can inspect separately.

Defaults follow the blog spec:
* ``min_cluster_size=5`` - a tag needs at least 5 near-duplicates to form
  a real environmental/agricultural category.
* ``min_samples=2`` - moderate conservatism; ambiguous singletons get
  pushed to noise rather than forming their own cluster.
* ``metric='euclidean'`` - safe now that the data is compressed via SVD
  (in the original 400k sparse space, euclidean would have been meaningless).
"""
from __future__ import annotations

from typing import Union

import numpy as np
from hdbscan import HDBSCAN


DEFAULT_MIN_CLUSTER_SIZE = 5
DEFAULT_MIN_SAMPLES = 2


def cluster_tags(
    dense: np.ndarray,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> np.ndarray:
    """Run HDBSCAN on *dense* (the SVD output) and return integer cluster labels.

    The returned array has shape ``(n_samples,)`` with values in ``[0, k)``
    for cluster members and ``-1`` for noise / anomalies.
    """
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    return clusterer.fit_predict(dense).astype(np.int64)
