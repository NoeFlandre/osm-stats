import numpy as np
import pytest

from src.core.features.cluster import (
    DEFAULT_MIN_CLUSTER_SIZE,
    DEFAULT_MIN_SAMPLES,
    cluster_tags,
)


# --- deterministic toy data ----------------------------------------------
#
# 30 rows in 5-D. Three clear groups of 8 (the 24 "structured" rows)
# plus 6 outliers (the "noise"). Euclidean distances are well separated.
def _toy_clusters():
    rng = np.random.default_rng(seed=0)
    n_per = 8
    # Three well-separated centers in 5-D; tight spread.
    centers = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [100.0, 100.0, 100.0, 100.0, 100.0],
            [-100.0, -100.0, -100.0, -100.0, -100.0],
        ]
    )
    in_cluster = np.vstack(
        [c + 0.5 * rng.standard_normal((n_per, 5)) for c in centers]
    )
    # Outliers placed far from every center.
    outliers = np.array(
        [
            [500.0, 500.0, 500.0, 500.0, 500.0],
            [-500.0, -500.0, -500.0, -500.0, -500.0],
            [500.0, -500.0, 500.0, -500.0, 500.0],
            [-500.0, 500.0, -500.0, 500.0, -500.0],
            [500.0, 500.0, -500.0, -500.0, 500.0],
            [-500.0, -500.0, 500.0, 500.0, -500.0],
        ]
    )
    X = np.vstack([in_cluster, outliers]).astype(np.float64)
    return X


# --- output shape and type ----------------------------------------------


def test_returns_integer_label_array():
    X = _toy_clusters()
    labels = cluster_tags(X)
    assert isinstance(labels, np.ndarray)
    assert labels.shape == (X.shape[0],)
    # Integer dtype is important for downstream indexing.
    assert np.issubdtype(labels.dtype, np.integer)


def test_output_length_matches_input():
    X = _toy_clusters()
    labels = cluster_tags(X)
    assert len(labels) == len(X)


# --- default parameters -------------------------------------------------


def test_default_min_cluster_size_is_5():
    assert DEFAULT_MIN_CLUSTER_SIZE == 5


def test_default_min_samples_is_2():
    assert DEFAULT_MIN_SAMPLES == 2


# --- behavior on the toy corpus -----------------------------------------


def test_three_dense_groups_are_recovered():
    X = _toy_clusters()
    labels = cluster_tags(X, min_cluster_size=5, min_samples=2)
    non_noise = labels[labels != -1]
    # Three groups of 8 -> exactly three distinct non-noise labels.
    assert len(set(non_noise)) == 3
    # Each cluster should have ~8 members.
    for cid in set(non_noise):
        assert np.sum(labels == cid) == 8


def test_far_outliers_become_noise_minus_one():
    X = _toy_clusters()
    labels = cluster_tags(X, min_cluster_size=5, min_samples=2)
    # The last 6 rows in the toy data are the outliers.
    assert np.all(labels[-6:] == -1)


# --- parameter overrides -----------------------------------------------


def test_min_cluster_size_is_honored():
    # A small min_cluster_size should not merge the 8-sized groups.
    X = _toy_clusters()
    labels_small = cluster_tags(X, min_cluster_size=4, min_samples=1)
    assert len(set(labels_small[labels_small != -1])) == 3


def test_min_samples_pushes_ambiguous_points_to_noise():
    # min_samples=5 is more conservative than min_samples=2: at least 5
    # neighbors must be within the core distance for a point to be a core
    # point. The toy 8-per-group points are dense enough to survive;
    # the outliers are not. We just check that the result is still
    # sensible (no negative cluster ids, three groups present).
    X = _toy_clusters()
    labels = cluster_tags(X, min_cluster_size=5, min_samples=5)
    # Three groups should still be detected, regardless of min_samples.
    assert len(set(labels[labels != -1])) == 3
    assert np.all(labels[-6:] == -1)
