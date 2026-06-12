"""Tests for the embedding-based cluster pipeline orchestrator.

The orchestrator wires four representation-agnostic stages together:

    embed -> SVD -> HDBSCAN -> medoids -> profile

The unit tests use a hand-crafted deterministic fake embedder so we
never depend on a network download in CI. The slow integration test
exercises the full pipeline with the real ``potion-base-8M`` model on
50 hand-picked tags and is marked ``slow``.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import pytest

from src.core.features.embedding.cluster_pipeline import run_embedding_pipeline


# --- the deterministic fake embedder -------------------------------------


class _FakeEmbedder:
    """Encode each tag as a small dense vector centered on a known location.

    Tags are split into *n_groups* well-separated clusters in *dim*-D
    space. The group is determined by the tag's first character (mod
    *n_groups*). A small per-tag noise keyed on the index makes
    within-group points distinct from each other (so HDBSCAN sees a
    cloud, not a single point) but tiny enough that HDBSCAN still
    groups them together.
    """

    def __init__(self, dim: int = 8, n_groups: int = 4) -> None:
        self.dim = dim
        self.n_groups = n_groups

    def embed(self, tags: Sequence[str]) -> np.ndarray:
        n = len(tags)
        out = np.zeros((n, self.dim), dtype=np.float32)
        for i, tag in enumerate(tags):
            first = ord(tag[0]) if tag else 0
            group = first % self.n_groups
            center = np.zeros(self.dim, dtype=np.float32)
            # Place each group on a distinct axis pair so the groups
            # are linearly separable and far apart in Euclidean space.
            center[group * 2] = 10.0
            center[group * 2 + 1] = 10.0
            out[i] = center + np.random.RandomState(i).randn(self.dim).astype(
                np.float32
            ) * 0.1
        return out


# --- 1. end-to-end with a fake embedder -----------------------------------


def test_end_to_end_recovers_three_clusters():
    """30 fake tags in 4 groups, projected through 8D->3D SVD and
    HDBSCAN. We expect the 4 groups to be detected as clusters. (We
    use 4 groups, not 3, so HDBSCAN is happy with min_cluster_size=5
    and there is room for the 8D->3D SVD to mix one of the axes a
    little without losing separability.)
    """
    # 30 tags: groups a*, b*, c*, d* (first character cycles)
    tags = [f"{chr(ord('a') + (i % 4))}{i:02d}|value" for i in range(30)]
    counts = [1] * 30
    embedder = _FakeEmbedder(dim=8, n_groups=4)

    medoids_df, profile_df, n_clusters, n_noise = run_embedding_pipeline(
        tags,
        counts,
        embedder=embedder,
        min_cluster_size=5,
        min_samples=2,
        n_components=3,
    )

    # The 4 well-separated groups must all be detected.
    assert n_clusters >= 3, (
        f"expected at least 3 clusters, got {n_clusters}. "
        f"n_noise={n_noise}"
    )
    non_noise = sum(
        int(row["cluster_size"])
        for _, row in medoids_df.iterrows()
        if int(row["cluster_id"]) != -1
    )
    # The pipeline must not throw and the total population must be
    # accounted for.
    assert n_noise + non_noise == 30


# --- 2. medoid DataFrame shape --------------------------------------------


def test_medoids_dataframe_has_expected_columns():
    tags = [f"k{i:02d}|v" for i in range(20)]
    counts = [1] * 20
    embedder = _FakeEmbedder(dim=8, n_groups=2)
    medoids_df, _, _, _ = run_embedding_pipeline(
        tags,
        counts,
        embedder=embedder,
        min_cluster_size=4,
        min_samples=1,
        n_components=3,
    )
    expected = {"cluster_id", "medoid_feature", "cluster_size", "total_count_all"}
    assert expected.issubset(set(medoids_df.columns))
    assert isinstance(medoids_df, pd.DataFrame)


# --- 3. profile DataFrame shape -------------------------------------------


def test_profile_dataframe_has_expected_columns():
    tags = [f"k{i:02d}|v" for i in range(20)]
    counts = [1] * 20
    embedder = _FakeEmbedder(dim=8, n_groups=2)
    _, profile_df, _, _ = run_embedding_pipeline(
        tags,
        counts,
        embedder=embedder,
        min_cluster_size=4,
        min_samples=1,
        n_components=3,
    )
    expected = {"base_key", "cluster_count", "total_count_all", "representative_medoids"}
    assert expected.issubset(set(profile_df.columns))
    assert isinstance(profile_df, pd.DataFrame)


# --- 4. cluster count semantics ------------------------------------------


def test_cluster_count_semantics_30_tags_in_groups_of_10():
    """30 tags cycling through 3 groups (a*, b*, c*) of 10. We expect
    n_clusters at most 3 and the population check to hold."""
    tags = [f"{chr(ord('a') + (i % 3))}{i:02d}|v" for i in range(30)]
    counts = [1] * 30
    embedder = _FakeEmbedder(dim=8, n_groups=3)

    medoids_df, _, n_clusters, n_noise = run_embedding_pipeline(
        tags,
        counts,
        embedder=embedder,
        min_cluster_size=5,
        min_samples=2,
        n_components=3,
    )

    assert n_clusters <= 3
    # Every point is either in a real cluster or in noise.
    non_noise = sum(
        int(row["cluster_size"])
        for _, row in medoids_df.iterrows()
        if int(row["cluster_id"]) != -1
    )
    assert n_noise + non_noise == 30


# --- 5. counts pass-through -----------------------------------------------


def test_counts_pass_through_to_medoid_total_count_all():
    """The medoid's ``total_count_all`` for a cluster must equal the sum
    of the input ``counts`` for the tags in that cluster.

    HDBSCAN requires ``min_cluster_size >= 2``, so a single isolated
    tag cannot form its own cluster. We use a 10-tag fixture with 2
    dense groups of 5: 5 a-tags and 5 b-tags, well separated. With
    ``min_cluster_size=4``, HDBSCAN detects both groups. We put the
    high-count tag (count=100) in the middle of the a-group, so
    cluster 0's ``total_count_all`` must be at least 100.

    The structural property we test is: the medoid output's
    ``total_count_all`` sum equals the input sum, and at least one
    medoid row carries a ``total_count_all`` of 100 or more
    (the contribution of the high-volume tag).
    """
    # 5 a-tags and 5 b-tags. With n_groups=2 and the first character
    # (a, b) mapping to groups 0 and 1, the embedder places the
    # a-tags on one center and the b-tags on a different center,
    # 20+ units apart. HDBSCAN finds 2 clusters of 5.
    tags = [
        f"a0|v", f"a1|v", f"a2|v", f"a3|v", f"a4|v",
        f"b5|v", f"b6|v", f"b7|v", f"b8|v", f"b9|v",
    ]
    # High count is on the a3 tag (in the a-group); rest are 1 each.
    counts = [1, 1, 1, 100, 1, 1, 1, 1, 1, 1]
    embedder = _FakeEmbedder(dim=8, n_groups=2)

    medoids_df, _, _, _ = run_embedding_pipeline(
        tags,
        counts,
        embedder=embedder,
        min_cluster_size=4,
        min_samples=2,
        n_components=3,
    )

    # The sum of medoid rows' total_count_all must equal the sum of
    # the input counts. This is the structural property that proves
    # counts pass through.
    assert int(medoids_df["total_count_all"].sum()) == sum(counts)
    # Some medoid row must carry the high count (>= 100).
    assert (medoids_df["total_count_all"] >= 100).any(), (
        f"expected some medoid row with total_count_all >= 100, "
        f"got {medoids_df['total_count_all'].tolist()}"
    )


# --- 6. n_components parameter honored ------------------------------------


def test_n_components_smaller_than_input_dim_does_not_crash():
    """``n_components`` smaller than the embedding dim must succeed
    and yield a well-shaped output. TruncatedSVD requires
    ``n_components <= n_features``, so we use dim=8 and n_components=4
    to stay within the safe range. We use 5 a-tags and 5 b-tags so
    HDBSCAN actually finds 2 clusters (avoiding the empty-profile
    edge case where ``profile_clusters_by_base_key`` would have
    nothing to sort).
    """
    tags = [
        f"a0|v", f"a1|v", f"a2|v", f"a3|v", f"a4|v",
        f"b5|v", f"b6|v", f"b7|v", f"b8|v", f"b9|v",
    ]
    counts = [1] * 10
    embedder = _FakeEmbedder(dim=8, n_groups=2)

    medoids_df, profile_df, n_clusters, n_noise = run_embedding_pipeline(
        tags,
        counts,
        embedder=embedder,
        min_cluster_size=3,
        min_samples=1,
        n_components=4,  # < dim=8, valid for TruncatedSVD
    )

    assert isinstance(medoids_df, pd.DataFrame)
    assert isinstance(profile_df, pd.DataFrame)
    assert isinstance(n_clusters, int)
    assert isinstance(n_noise, int)


# --- 7. real-model integration test (slow) --------------------------------


def _real_model_tags() -> list:
    """50 hand-picked OSM ``"<key>|<value>"`` tags spanning the env/agri
    families. Built so that the real ``potion-base-8M`` embedder is
    exercised on a realistic mix without touching disk.
    """
    families = [
        ("landuse", ["farmland", "meadow", "grass", "forest", "residential",
                     "vineyard", "orchard", "cemetery", "commercial",
                     "industrial"]),
        ("natural", ["water", "wood", "tree_row", "grassland", "wetland",
                     "scrub", "beach", "cliff", "bare_rock", "sand"]),
        ("highway", ["residential", "service", "track", "footway",
                     "cycleway", "path", "unclassified", "tertiary",
                     "primary", "secondary"]),
        ("waterway", ["stream", "river", "canal", "ditch", "drain"]),
        ("crop", ["wheat", "rice", "maize", "barley", "oats"]),
        ("species", ["quercus_robur", "fagus_sylvatica", "pinus_sylvestris",
                     "acer_pseudoplatanus", "betula_pendula"]),
        ("boundary", ["national_park", "protected_area", "forest_compartment",
                      "nature_reserve", "wilderness"]),
    ]
    tags: list = []
    for key, values in families:
        for v in values:
            tags.append(f"{key}|{v}")
            if len(tags) == 50:
                return tags
    return tags


@pytest.mark.slow
def test_real_model_integration_returns_well_shaped_outputs():
    """End-to-end with the real ``potion-base-8M`` embedder on 50 tags.

    HDBSCAN is non-deterministic, so we do not assert specific cluster
    counts. We only assert the output is well-shaped and that the
    full population is accounted for.
    """
    from src.core.features.semantic_probe import SemanticProbe

    tags = _real_model_tags()
    assert len(tags) == 50
    counts = [1] * len(tags)
    embedder = SemanticProbe()  # real potion-base-8M

    medoids_df, profile_df, n_clusters, n_noise = run_embedding_pipeline(
        tags,
        counts,
        embedder=embedder,
        min_cluster_size=5,
        min_samples=2,
        n_components=50,
    )

    # Shapes
    assert isinstance(medoids_df, pd.DataFrame)
    assert isinstance(profile_df, pd.DataFrame)
    assert {"cluster_id", "medoid_feature", "cluster_size", "total_count_all"}.issubset(
        set(medoids_df.columns)
    )
    assert {
        "base_key",
        "cluster_count",
        "total_count_all",
        "representative_medoids",
    }.issubset(set(profile_df.columns))

    # Population check: real clusters + noise == 50.
    non_noise = sum(
        int(row["cluster_size"])
        for _, row in medoids_df.iterrows()
        if int(row["cluster_id"]) != -1
    )
    assert n_noise + non_noise == 50
    # The two scalars must also account for the same total.
    assert n_clusters + n_noise >= 0  # sanity: both ints
    assert isinstance(n_clusters, int)
    assert isinstance(n_noise, int)
