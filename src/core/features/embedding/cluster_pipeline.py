"""Embedding-based orchestrator: embed -> SVD -> HDBSCAN -> medoids -> profile.

This is the embedding-based parallel pipeline. It is shape-equivalent to
``scripts/profile_clusters.py`` (the TF-IDF + clustering orchestrator) but
uses a static vector model (``potion-base-8M`` by default) instead of
character n-gram TF-IDF to project each ``"<key>|<value>"`` tag into a
dense vector.

The downstream stages are the same as in the TF-IDF pipeline because they
are representation-agnostic:

* :mod:`src.core.features.reduce` - TruncatedSVD projects the raw 384-d
  embedding down to a low-dimensional dense space (50-d by default). The
  SVD step is necessary because HDBSCAN is unreliable in very high
  dimensions (curse of dimensionality); this matches the TF-IDF pipeline,
  which also clusters on the SVD output.
* :mod:`src.core.features.cluster` - HDBSCAN groups dense regions and
  pushes isolated points into a noise bucket labelled -1.
* :mod:`src.core.features.medoids` - one representative tag per cluster.
* :mod:`src.core.features.profile` - one row per base-key family.

This module reuses all four of those without modification. The only new
piece is the :func:`run_embedding_pipeline` entry point that wires them
together in the embedding order.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd

from src.core.features.cluster import (
    DEFAULT_MIN_CLUSTER_SIZE,
    DEFAULT_MIN_SAMPLES,
    cluster_tags,
)
from src.core.features.medoids import compute_cluster_medoids
from src.core.features.profile import profile_clusters_by_base_key
from src.core.features.reduce import DEFAULT_N_COMPONENTS, reduce_dimensions


def run_embedding_pipeline(
    tags: Sequence[str],
    counts: Sequence[int],
    *,
    embedder: object,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    n_components: int = DEFAULT_N_COMPONENTS,
) -> Tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """Run embed -> SVD -> HDBSCAN -> medoids -> profile.

    Parameters
    ----------
    tags:
        Sequence of standardized ``"<key>|<value>"`` tag strings.
    counts:
        Sequence of occurrence counts, the same length as *tags*.
    embedder:
        Any object with an ``.embed(tags) -> np.ndarray`` method
        producing a ``(len(tags), dim)`` dense array. The real
        :class:`src.core.features.semantic_probe.SemanticProbe` works;
        any duck-typed equivalent works too.
    min_cluster_size:
        Forwarded to HDBSCAN. Defaults to 5 (the TF-IDF default).
    min_samples:
        Forwarded to HDBSCAN. Defaults to 2 (the TF-IDF default).
    n_components:
        Forwarded to the SVD reduction step. Defaults to 50 (the
        TF-IDF default). ``TruncatedSVD`` clamps to
        ``min(n_components, n_features)`` automatically, so passing a
        value larger than the embedding dim is safe (it just skips
        the SVD step in spirit).

    Returns
    -------
    ``(medoids_df, profile_df, n_clusters, n_noise)`` where:

    * ``medoids_df`` is the output of :func:`compute_cluster_medoids`
      with one row per cluster (and one extra row for the noise
      bucket, label -1, when noise is present).
    * ``profile_df`` is the output of
      :func:`profile_clusters_by_base_key` with one row per base-key
      family (noise excluded).
    * ``n_clusters`` is the count of distinct non-negative cluster
      ids in the HDBSCAN output.
    * ``n_noise`` is the count of points labelled -1.

    Notes
    -----
    The pipeline is non-deterministic: ``TruncatedSVD`` and ``HDBSCAN``
    both use randomness. The blog post treats this as acceptable; the
    test suite works around it by exercising the pipeline on a small
    fake embedder and asserting structural properties (shapes,
    counts) rather than exact cluster assignments.
    """
    tags_list = list(tags)
    counts_list = list(counts)
    if len(tags_list) != len(counts_list):
        raise ValueError(
            "run_embedding_pipeline: tags and counts must have the same length"
        )

    vecs = np.asarray(embedder.embed(tags_list))
    dense = reduce_dimensions(vecs, n_components=n_components)
    labels = cluster_tags(
        dense, min_cluster_size=min_cluster_size, min_samples=min_samples
    )
    medoids_df = compute_cluster_medoids(
        features=tags_list, dense=dense, labels=labels, counts=counts_list
    )
    profile_df = profile_clusters_by_base_key(medoids_df, top_n=5)

    distinct = {int(x) for x in labels.tolist()}
    n_clusters = len(distinct - {-1})
    n_noise = int((labels == -1).sum())

    return medoids_df, profile_df, n_clusters, n_noise
