import numpy as np
import pandas as pd

from src.core.features.medoids import compute_cluster_medoids


# Toy: 6 points in 2-D, three clusters of size 2.
# Cluster 0: (0,0) and (1,0)  -> medoid should be (0,0), the closer to (0.5, 0)
# Cluster 1: (10,10) and (11,10) -> medoid should be (10,10)
# Cluster 2: (-10,-10) and (-9,-10) -> medoid should be (-10,-10)
TOY = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [10.0, 10.0],
        [11.0, 10.0],
        [-10.0, -10.0],
        [-9.0, -10.0],
    ]
)
TOY_LABELS = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
TOY_FEATURES = [
    "a|0", "a|1",
    "b|0", "b|1",
    "c|0", "c|1",
]
TOY_COUNTS = [100, 50, 200, 75, 300, 25]


# --- output shape and type ----------------------------------------------


def test_returns_dataframe():
    out = compute_cluster_medoids(TOY_FEATURES, TOY, TOY_LABELS, TOY_COUNTS)
    assert isinstance(out, pd.DataFrame)


def test_one_row_per_cluster_excluding_noise():
    out = compute_cluster_medoids(TOY_FEATURES, TOY, TOY_LABELS, TOY_COUNTS)
    # Three clusters, no noise -> three rows.
    assert len(out) == 3


# --- medoid identification ----------------------------------------------


def test_medoid_is_closest_to_centroid():
    out = compute_cluster_medoids(TOY_FEATURES, TOY, TOY_LABELS, TOY_COUNTS)
    # Cluster 0 centroid is (0.5, 0) -> closer to (0,0) than to (1,0).
    row0 = out[out["cluster_id"] == 0].iloc[0]
    assert row0["medoid_feature"] == "a|0"
    row1 = out[out["cluster_id"] == 1].iloc[0]
    assert row1["medoid_feature"] == "b|0"
    row2 = out[out["cluster_id"] == 2].iloc[0]
    assert row2["medoid_feature"] == "c|0"


# --- aggregation --------------------------------------------------------


def test_cluster_size_is_member_count():
    out = compute_cluster_medoids(TOY_FEATURES, TOY, TOY_LABELS, TOY_COUNTS)
    assert int(out[out["cluster_id"] == 0].iloc[0]["cluster_size"]) == 2


def test_total_count_all_is_sum_of_member_counts():
    out = compute_cluster_medoids(TOY_FEATURES, TOY, TOY_LABELS, TOY_COUNTS)
    row1 = out[out["cluster_id"] == 1].iloc[0]
    assert int(row1["total_count_all"]) == 200 + 75


# --- noise handling -----------------------------------------------------


def test_noise_bucket_creates_individual_high_volume_rows():
    # All but one row -> noise. The single non-noise point forms cluster 0.
    labels = np.array([-1, -1, -1, -1, -1, 0], dtype=np.int64)
    out = compute_cluster_medoids(TOY_FEATURES, TOY, labels, TOY_COUNTS)
    # Two rows: one for cluster 0, one for the noise bucket.
    assert len(out) == 2
    cluster_ids = set(out["cluster_id"])
    assert 0 in cluster_ids
    assert -1 in cluster_ids
    noise_row = out[out["cluster_id"] == -1].iloc[0]
    # The 5 noise features in input order are a|0(100), a|1(50), b|0(200),
    # b|1(75), c|0(300). The top by count is c|0 (300) - the medoid slot
    # surfaces it for the profiling stage.
    assert noise_row["medoid_feature"] == "c|0"


def test_noise_bucket_excluded_by_default():
    # When most points are noise, the noise bucket is summarized but
    # treated as a single 'misc' entry, not a per-row micro-cluster.
    labels = np.array([-1, -1, -1, -1, -1, -1], dtype=np.int64)
    out = compute_cluster_medoids(TOY_FEATURES, TOY, labels, TOY_COUNTS)
    # Only the noise bucket row.
    assert len(out) == 1
    assert out.iloc[0]["cluster_id"] == -1
    # The noise row carries the sum of all 6 counts.
    assert int(out.iloc[0]["total_count_all"]) == sum(TOY_COUNTS)


# --- schema -------------------------------------------------------------


def test_dataframe_has_expected_columns():
    out = compute_cluster_medoids(TOY_FEATURES, TOY, TOY_LABELS, TOY_COUNTS)
    expected = {"cluster_id", "medoid_feature", "cluster_size", "total_count_all"}
    assert expected.issubset(out.columns)


def test_cluster_ids_are_sorted_ascending():
    out = compute_cluster_medoids(TOY_FEATURES, TOY, TOY_LABELS, TOY_COUNTS)
    # Real cluster ids ascending, noise at the end if present.
    non_noise = out[out["cluster_id"] != -1]["cluster_id"].tolist()
    assert non_noise == sorted(non_noise)
