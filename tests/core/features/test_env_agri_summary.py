"""Stats over the env/agri breakdown: row count, cluster count,
total occurrences, top base key, etc.

The numbers come straight from the per-cluster medoid file, so the
totals are sums of the per-cluster counts (not the global
'count_all' from the underlying taginfo DB - that would mix cluster
members with non-cluster noise).
"""
from src.core.features.env_agri_summary import (
    env_agri_summary,
    render_env_agri_summary_markdown,
)
from src.core.features.env_agri_whitelist import ENVI_AGRI_BASE_KEYS


# --- env_agri_summary --------------------------------------------------


def test_summary_returns_dict():
    s = env_agri_summary()
    assert isinstance(s, dict)


def test_summary_has_expected_keys():
    s = env_agri_summary()
    for k in (
        "n_base_keys",
        "n_clusters",
        "total_occurrences",
        "top_base_key",
        "top_base_key_occurrences",
    ):
        assert k in s, f"missing key: {k}"


def test_summary_n_base_keys_matches_whitelist():
    s = env_agri_summary()
    assert s["n_base_keys"] == len(ENVI_AGRI_BASE_KEYS)


def test_summary_n_clusters_is_positive_integer():
    s = env_agri_summary()
    assert isinstance(s["n_clusters"], int)
    assert s["n_clusters"] > 0


def test_summary_total_occurrences_is_positive_integer():
    s = env_agri_summary()
    assert isinstance(s["total_occurrences"], int)
    assert s["total_occurrences"] > 0


def test_summary_top_base_key_is_in_whitelist():
    s = env_agri_summary()
    assert s["top_base_key"] in ENVI_AGRI_BASE_KEYS


def test_summary_top_base_key_occurrences_matches_top_base_key():
    s = env_agri_summary()
    # The top base key's occurrences must equal the highest family
    # total in the breakdown.
    from src.core.features.breakdown import env_agri_breakdown_df

    df = env_agri_breakdown_df()
    by_key = df.groupby("base_key")["total_count_all"].sum()
    assert s["top_base_key_occurrences"] == int(by_key.max())


# --- render_env_agri_summary_markdown ----------------------------------


def test_render_returns_string():
    md = render_env_agri_summary_markdown()
    assert isinstance(md, str)


def test_render_includes_every_stat():
    md = render_env_agri_summary_markdown()
    assert "base keys" in md.lower()
    assert "clusters" in md.lower()
    assert "occurrences" in md.lower()
    assert "top base key" in md.lower()
