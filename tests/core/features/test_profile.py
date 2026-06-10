import pandas as pd

from src.core.features.profile import profile_clusters_by_base_key


def _medoids_df():
    return pd.DataFrame(
        {
            "cluster_id": [0, 1, 2, 3, 4, 5, 6, -1],
            "medoid_feature": [
                "addr:street|hauptstraße",
                "addr:city|berlin",
                "addr:postcode|10115",
                "landuse|farmland",
                "landuse|forest",
                "natural|water",
                "building|house",
                "",
            ],
            "cluster_size": [10, 8, 12, 6, 5, 7, 100, 50],
            "total_count_all": [
                1_000_000,
                500_000,
                300_000,
                800_000,
                200_000,
                400_000,
                50_000_000,
                1_000,
            ],
        }
    )


# --- output shape and type ----------------------------------------------


def test_returns_dataframe():
    out = profile_clusters_by_base_key(_medoids_df(), top_n=3)
    assert isinstance(out, pd.DataFrame)


def test_one_row_per_base_key():
    out = profile_clusters_by_base_key(_medoids_df(), top_n=3)
    # 4 unique base keys: addr, landuse, natural, building (noise excluded).
    assert len(out) == 4


# --- aggregation --------------------------------------------------------


def test_cluster_count_sums_member_clusters():
    out = profile_clusters_by_base_key(_medoids_df(), top_n=3)
    addr = out[out["base_key"] == "addr"].iloc[0]
    assert int(addr["cluster_count"]) == 3  # clusters 0, 1, 2


def test_total_count_all_sums_member_counts():
    out = profile_clusters_by_base_key(_medoids_df(), top_n=3)
    addr = out[out["base_key"] == "addr"].iloc[0]
    assert int(addr["total_count_all"]) == 1_000_000 + 500_000 + 300_000


def test_representative_medoids_capped_at_top_n():
    out = profile_clusters_by_base_key(_medoids_df(), top_n=3)
    building = out[out["base_key"] == "building"].iloc[0]
    # Only one building cluster, so only one medoid.
    assert building["representative_medoids"] == "building|house"


def test_representative_medoids_sorted_by_count_desc():
    out = profile_clusters_by_base_key(_medoids_df(), top_n=3)
    addr = out[out["base_key"] == "addr"].iloc[0]
    medoids = addr["representative_medoids"].split("; ")
    # 1M > 500k > 300k -> order preserved.
    assert medoids[0] == "addr:street|hauptstraße"
    assert medoids[1] == "addr:city|berlin"
    assert medoids[2] == "addr:postcode|10115"


# --- noise exclusion ---------------------------------------------------


def test_noise_bucket_excluded_from_profile():
    out = profile_clusters_by_base_key(_medoids_df(), top_n=3)
    assert "noise" not in set(out["base_key"])


def test_noise_row_with_empty_medoid_excluded():
    df = _medoids_df()
    out = profile_clusters_by_base_key(df, top_n=3)
    # 4 base keys, none of them "noise".
    assert len(out) == 4
    assert (out["base_key"] == "noise").sum() == 0


# --- sort order --------------------------------------------------------


def test_output_sorted_by_total_count_all_desc():
    out = profile_clusters_by_base_key(_medoids_df(), top_n=3)
    totals = out["total_count_all"].tolist()
    assert totals == sorted(totals, reverse=True)
    # Building is by far the largest in this toy set.
    assert out.iloc[0]["base_key"] == "building"


# --- schema -------------------------------------------------------------


def test_dataframe_has_expected_columns():
    out = profile_clusters_by_base_key(_medoids_df(), top_n=3)
    expected = {"base_key", "cluster_count", "total_count_all", "representative_medoids"}
    assert expected.issubset(out.columns)
