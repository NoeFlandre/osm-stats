"""Tests for the TF-IDF vs Embeddings comparison module.

The comparison module is a *pure* module that reads the two pipelines'
persisted artifacts (markdown profile + medoid CSV) and produces a
side-by-side Markdown report. It must NOT call into the TF-IDF
breakdown module (``src.core.features.breakdown``) because that
module reads from the TF-IDF medoid file only.

TDD: these tests are written first, then ``comparison.py`` is
implemented to make them pass.
"""
from __future__ import annotations

from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from src.core.features.embedding import comparison as cmp


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


def _make_tfidf_profile_md() -> str:
    return (
        "| base_key | cluster_count | total_count_all | representative_medoids |\n"
        "| --- | --- | --- | --- |\n"
        "| addr | 4,790 | 465,963,996 | addr:country\\|af; addr:state\\|dc |\n"
        "| source | 668 | 231,299,922 | source\\|dps; source\\|esri |\n"
        "| landuse | 2 | 30,325,838 | landuse\\|orchard; landuse\\|farmland |\n"
    )


def _make_tfidf_medoids_csv() -> str:
    return (
        "cluster_id,medoid_feature,cluster_size,total_count_all\n"
        "0,landuse|orchard,5,1000\n"
        "1,landuse|farmland,4,800\n"
        "2,addr:country|af,10,5000\n"
        "3,natural|water,3,600\n"
    )


# ---------------------------------------------------------------------
# load_profile_from_markdown
# ---------------------------------------------------------------------


def test_load_profile_from_markdown_roundtrip(tmp_path: Path):
    md = _make_tfidf_profile_md()
    p = tmp_path / "profile.md"
    p.write_text(md)

    df = cmp.load_profile_from_markdown(p)
    assert isinstance(df, pd.DataFrame)
    assert list(df["base_key"]) == ["addr", "source", "landuse"]
    assert list(df["cluster_count"]) == [4790, 668, 2]
    assert list(df["total_count_all"]) == [465963996, 231299922, 30325838]
    # The medoid cells contain escaped pipes; loading should unescape them.
    assert df.iloc[0]["representative_medoids"] == "addr:country|af; addr:state|dc"


def test_load_profile_from_markdown_handles_escaped_pipes(tmp_path: Path):
    md = (
        "| base_key | cluster_count | total_count_all | representative_medoids |\n"
        "| --- | --- | --- | --- |\n"
        "| landuse | 2 | 30,325,838 | landuse\\|orchard; landuse\\|farmland |\n"
    )
    p = tmp_path / "profile.md"
    p.write_text(md)

    df = cmp.load_profile_from_markdown(p)
    assert df.iloc[0]["representative_medoids"] == "landuse|orchard; landuse|farmland"


def test_load_profile_from_markdown_handles_comma_separated_numbers(tmp_path: Path):
    md = (
        "| base_key | cluster_count | total_count_all | representative_medoids |\n"
        "| --- | --- | --- | --- |\n"
        "| landuse | 4,790 | 1,234,567 | landuse\\|orchard |\n"
    )
    p = tmp_path / "profile.md"
    p.write_text(md)

    df = cmp.load_profile_from_markdown(p)
    assert df.iloc[0]["cluster_count"] == 4790
    assert int(df.iloc[0]["total_count_all"]) == 1234567


# ---------------------------------------------------------------------
# load_medoids_from_csv
# ---------------------------------------------------------------------


def test_load_medoids_from_csv_basic(tmp_path: Path):
    p = tmp_path / "medoids.csv"
    p.write_text(_make_tfidf_medoids_csv())

    df = cmp.load_medoids_from_csv(p)
    assert isinstance(df, pd.DataFrame)
    expected = {"cluster_id", "medoid_feature", "cluster_size", "total_count_all"}
    assert expected.issubset(set(df.columns))
    assert len(df) == 4
    assert int(df.iloc[0]["cluster_id"]) == 0


# ---------------------------------------------------------------------
# build_comparison
# ---------------------------------------------------------------------


def test_build_comparison_outer_joins_on_base_key():
    tfidf = pd.DataFrame(
        {
            "base_key": ["addr", "landuse"],
            "cluster_count": [10, 5],
            "total_count_all": [1000, 500],
            "representative_medoids": ["a", "b"],
        }
    )
    embed = pd.DataFrame(
        {
            "base_key": ["landuse", "natural"],
            "cluster_count": [6, 3],
            "total_count_all": [600, 200],
            "representative_medoids": ["c", "d"],
        }
    )
    out = cmp.build_comparison(tfidf, embed)
    keys = set(out["base_key"])
    assert keys == {"addr", "landuse", "natural"}
    assert len(out) == 3


def test_build_comparison_computes_deltas_correctly():
    tfidf = pd.DataFrame(
        {
            "base_key": ["landuse"],
            "cluster_count": [5],
            "total_count_all": [1000],
            "representative_medoids": ["a"],
        }
    )
    embed = pd.DataFrame(
        {
            "base_key": ["landuse"],
            "cluster_count": [8],
            "total_count_all": [1500],
            "representative_medoids": ["b"],
        }
    )
    out = cmp.build_comparison(tfidf, embed)
    row = out.iloc[0]
    assert int(row["tfidf_cluster_count"]) == 5
    assert int(row["embedding_cluster_count"]) == 8
    assert int(row["cluster_count_delta"]) == 3
    assert int(row["tfidf_total_count_all"]) == 1000
    assert int(row["embedding_total_count_all"]) == 1500
    assert int(row["total_count_all_delta"]) == 500
    # 500 / 1000 * 100 = 50.0
    assert float(row["total_count_all_pct_change"]) == 50.0


def test_build_comparison_sorts_by_abs_delta():
    tfidf = pd.DataFrame(
        {
            "base_key": ["A", "B"],
            "cluster_count": [10, 10],
            "total_count_all": [1000, 1000],
            "representative_medoids": ["a", "b"],
        }
    )
    embed = pd.DataFrame(
        {
            "base_key": ["A", "B"],
            "cluster_count": [11, 50],
            "total_count_all": [1010, 5000],
            "representative_medoids": ["a", "b"],
        }
    )
    out = cmp.build_comparison(tfidf, embed)
    # A's delta is 10, B's is 4000. B should come first.
    assert out.iloc[0]["base_key"] == "B"
    assert out.iloc[1]["base_key"] == "A"


# ---------------------------------------------------------------------
# build_env_agri_comparison
# ---------------------------------------------------------------------


def test_build_env_agri_comparison_aggregates_by_base_key():
    tfidf_bd = pd.DataFrame(
        {
            "base_key": ["landuse", "landuse", "natural"],
            "cluster_id": [1, 2, 3],
            "medoid": ["a", "b", "c"],
            "cluster_size": [10, 20, 5],
            "total_count_all": [100, 200, 50],
        }
    )
    embed_bd = pd.DataFrame(
        {
            "base_key": ["landuse", "landuse"],
            "cluster_id": [10, 11],
            "medoid": ["x", "y"],
            "cluster_size": [4, 6],
            "total_count_all": [80, 120],
        }
    )
    out = cmp.build_env_agri_comparison(tfidf_bd, embed_bd)
    by_key = {row["base_key"]: row for _, row in out.iterrows()}

    # landuse appears in both -> 2 distinct cluster_ids in tfidf, 2 in embed.
    landuse = by_key["landuse"]
    assert int(landuse["tfidf_clusters"]) == 2
    assert int(landuse["embedding_clusters"]) == 2
    assert int(landuse["tfidf_occurrences"]) == 300
    assert int(landuse["embedding_occurrences"]) == 200
    assert int(landuse["clusters_delta"]) == 0
    assert int(landuse["occurrences_delta"]) == -100

    # natural appears only in tfidf.
    natural = by_key["natural"]
    assert int(natural["tfidf_clusters"]) == 1
    assert int(natural["embedding_clusters"]) == 0
    assert int(natural["tfidf_occurrences"]) == 50
    assert int(natural["embedding_occurrences"]) == 0


# ---------------------------------------------------------------------
# build_env_agri_breakdown_from_medoids
# ---------------------------------------------------------------------


def test_build_env_agri_breakdown_from_medoids_filters_to_whitelist(tmp_path: Path):
    csv_text = (
        "cluster_id,medoid_feature,cluster_size,total_count_all\n"
        "0,landuse|orchard,5,1000\n"
        "1,landuse|farmland,4,800\n"
        "2,addr:country|af,10,5000\n"
        "3,natural|water,3,600\n"
        "4,highway|residential,7,2000\n"
    )
    p = tmp_path / "medoids.csv"
    p.write_text(csv_text)

    df = cmp.build_env_agri_breakdown_from_medoids(p)
    # Only landuse and natural are in the env/agri whitelist; the others
    # (addr, highway) are filtered out.
    assert set(df["base_key"]) <= set(cmp.ENVI_AGRI_BASE_KEYS)
    assert sorted(df["base_key"].unique().tolist()) == ["landuse", "natural"]
    # Schema must include the five required columns.
    expected_cols = ["base_key", "cluster_id", "medoid", "cluster_size", "total_count_all"]
    assert list(df.columns) == expected_cols


def test_build_env_agri_breakdown_from_medoids_missing_file_raises(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        cmp.build_env_agri_breakdown_from_medoids(missing)


# ---------------------------------------------------------------------
# render_comparison_markdown
# ---------------------------------------------------------------------


def test_render_comparison_markdown_contains_all_sections():
    comparison_df = pd.DataFrame(
        {
            "base_key": ["landuse", "natural"],
            "tfidf_cluster_count": [5, 3],
            "embedding_cluster_count": [6, 3],
            "cluster_count_delta": [1, 0],
            "tfidf_total_count_all": [1000, 500],
            "embedding_total_count_all": [1200, 500],
            "total_count_all_delta": [200, 0],
            "total_count_all_pct_change": [20.0, 0.0],
        }
    )
    env_agri_df = pd.DataFrame(
        {
            "base_key": ["landuse"],
            "tfidf_clusters": [5],
            "embedding_clusters": [6],
            "clusters_delta": [1],
            "tfidf_occurrences": [1000],
            "embedding_occurrences": [1200],
            "occurrences_delta": [200],
        }
    )

    md = cmp.render_comparison_markdown(comparison_df, env_agri_df)
    assert isinstance(md, str)
    # Section headers
    assert "# Pipeline Comparison: TF-IDF vs Embeddings" in md
    assert "## Global stats" in md
    assert "## Top 20 base keys by combined volume" in md
    assert "## Env/Agri whitelist comparison" in md
    # Both base keys are rendered in the top-N table.
    assert "| landuse |" in md
    assert "| natural |" in md
    # The env/agri row appears.
    assert "landuse" in md
    # Headers of each table are present.
    assert "| metric | tfidf | embeddings | delta |" in md
    assert (
        "| base_key | tfidf_clusters | embedding_clusters | clusters_delta |" in md
    )
    assert (
        "| base_key | tfidf_clusters | embedding_clusters | clusters_delta |"
        " tfidf_occurrences | embedding_occurrences | occurrences_delta |" in md
    )


def test_render_comparison_markdown_escapes_pipes():
    """A base key or medoid containing '|' must be escaped to '\\|'
    in the rendered Markdown so the table stays parseable."""
    # The medoid in the profile contains a '|'.
    tfidf_profile = pd.DataFrame(
        {
            "base_key": ["landuse"],
            "cluster_count": [5],
            "total_count_all": [1000],
            "representative_medoids": ["landuse|orchard"],
        }
    )
    embed_profile = pd.DataFrame(
        {
            "base_key": ["landuse"],
            "cluster_count": [5],
            "total_count_all": [1000],
            "representative_medoids": ["landuse|orchard"],
        }
    )
    comparison_df = cmp.build_comparison(tfidf_profile, embed_profile)
    env_agri_df = pd.DataFrame(
        columns=[
            "base_key",
            "tfidf_clusters",
            "embedding_clusters",
            "clusters_delta",
            "tfidf_occurrences",
            "embedding_occurrences",
            "occurrences_delta",
        ]
    )
    md = cmp.render_comparison_markdown(comparison_df, env_agri_df)
    # The most important regression check: no un-escaped '|' should
    # create an extra column when split. We accept rows of 6 cells
    # (global-stats: 4 columns + leading + trailing), 9 cells
    # (env/agri: 7 columns + leading + trailing), or 10 cells
    # (top-20: 8 columns + leading + trailing).
    for line in md.splitlines():
        if not line.startswith("|"):
            continue
        # Replace escaped pipes with a placeholder, then split.
        cleaned = line.replace("\\|", "\x00")
        cells = [c.strip() for c in cleaned.split("|")]
        assert len(cells) in (6, 9, 10), f"malformed row: {line!r}"


# ---------------------------------------------------------------------
# Regression: comparison does not depend on the TF-IDF breakdown module
# ---------------------------------------------------------------------


def test_pipeline_comparison_does_not_read_tfidf_breakdown_module():
    """A regression guard: none of the comparison functions should
    call ``env_agri_breakdown_df`` (the TF-IDF breakdown module).
    The comparison reads from CSV files only."""
    with mock.patch(
        "src.core.features.breakdown.env_agri_breakdown_df"
    ) as mocked:
        # Exercise every public function. None of them should hit the
        # TF-IDF breakdown module.
        tfidf_profile = pd.DataFrame(
            {
                "base_key": ["landuse"],
                "cluster_count": [5],
                "total_count_all": [1000],
                "representative_medoids": ["landuse|orchard"],
            }
        )
        embed_profile = pd.DataFrame(
            {
                "base_key": ["landuse"],
                "cluster_count": [5],
                "total_count_all": [1000],
                "representative_medoids": ["landuse|orchard"],
            }
        )
        tfidf_bd = pd.DataFrame(
            {
                "base_key": ["landuse"],
                "cluster_id": [1],
                "medoid": ["x"],
                "cluster_size": [10],
                "total_count_all": [100],
            }
        )
        embed_bd = tfidf_bd.copy()
        cmp.build_comparison(tfidf_profile, embed_profile)
        cmp.build_env_agri_comparison(tfidf_bd, embed_bd)
        # The CSV path uses the env/agri whitelist, not the breakdown module.
        # The markdown render reads from the in-memory DataFrames.
        cmp.render_comparison_markdown(
            cmp.build_comparison(tfidf_profile, embed_profile),
            cmp.build_env_agri_comparison(tfidf_bd, embed_bd),
        )
        mocked.assert_not_called()
