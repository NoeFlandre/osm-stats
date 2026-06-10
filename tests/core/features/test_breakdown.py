"""The env/agri breakdown must produce one row per representative
cluster medoid for the whitelisted base keys, with cluster_id and the
medoid feature for traceability.

The representative medoids come from ``output/cluster_profile.md``,
where the cluster-profile step emits the top-N medoids per base key
sorted by ``total_count_all`` descending. The breakdown expands those
into a flat per-cluster view for the env/agri whitelist.
"""
from pathlib import Path
import re

from src.core.features.env_agri_whitelist import ENVI_AGRI_BASE_KEYS
from src.core.features.breakdown import (
    env_agri_breakdown_df,
    render_breakdown_markdown,
)


def _profile_rows():
    md = Path("output/cluster_profile.md").read_text()
    rows = []
    placeholder = "\x00PIPE\x00"
    for line in md.splitlines():
        if not line.startswith("| ") or "base_key" in line or line.startswith("| ---"):
            continue
        cleaned = line.replace("\\|", placeholder)
        cells = [c.strip() for c in cleaned.split("|")]
        cells = [c.replace(placeholder, "\\|") for c in cells]
        if len(cells) < 5:
            continue
        bk = cells[1]
        if not bk or bk == "---":
            continue
        cc = int(cells[2].replace(",", ""))
        tc = int(cells[3].replace(",", ""))
        meds = [m for m in cells[4].split("; ") if m]
        rows.append((bk, cc, tc, meds))
    return rows


# --- env_agri_breakdown_df ---------------------------------------------


def test_breakdown_returns_dataframe():
    df = env_agri_breakdown_df()
    assert not df.empty
    assert list(df.columns) == ["base_key", "cluster_id", "medoid", "cluster_size", "total_count_all"]


def test_breakdown_only_contains_whitelisted_base_keys():
    df = env_agri_breakdown_df()
    assert set(df["base_key"]) <= ENVI_AGRI_BASE_KEYS

def test_breakdown_cluster_ids_are_unique():
    df = env_agri_breakdown_df()
    assert df["cluster_id"].is_unique

def test_breakdown_total_clusters_matches_profile_aggregate():
    df = env_agri_breakdown_df()
    profile_rows = _profile_rows()
    # The profile emits the top-N medoids per base key. The breakdown
    # expands those into one row per medoid, so the row count equals
    # the sum of representative medoids over the whitelisted base keys.
    expected = sum(len(meds) for bk, _, _, meds in profile_rows if bk in ENVI_AGRI_BASE_KEYS)
    assert len(df) == expected

def test_breakdown_total_count_all_does_not_exceed_profile_aggregate():
    # The breakdown emits the per-base-key total_count_all on every
    # medoid row (we don't have per-cluster counts on disk). Summing
    # across medoids for a base key therefore over-counts, but each
    # individual row's value must match the per-base-key total.
    df = env_agri_breakdown_df()
    profile_rows = _profile_rows()
    by_key = {bk: tc for bk, _, tc, _ in profile_rows}
    for _, row in df.iterrows():
        assert int(row["total_count_all"]) == by_key[row["base_key"]]


# --- render_breakdown_markdown -----------------------------------------


def test_render_returns_string():
    md = render_breakdown_markdown(env_agri_breakdown_df())
    assert isinstance(md, str)


def test_render_has_header_and_separator():
    md = render_breakdown_markdown(env_agri_breakdown_df())
    assert "| base_key |" in md
    assert "| --- |" in md


def test_render_includes_one_row_per_cluster():
    df = env_agri_breakdown_df()
    md = render_breakdown_markdown(df)
    body_rows = [ln for ln in md.splitlines() if ln.startswith("| ") and "base_key" not in ln and "---" not in ln]
    assert len(body_rows) == len(df)


def test_render_includes_medoid_for_every_cluster():
    df = env_agri_breakdown_df()
    md = render_breakdown_markdown(df)
    for medoid in df["medoid"]:
        # The medoid in the rendered table has its '|' escaped.
        # Empty medoids (some rows have no medoids) are skipped here.
        if not medoid:
            continue
        # Source medoids are already in the form 'key\|value'. The
        # renderer's escape preserves existing escapes and only escapes
        # un-escaped pipes, so the output contains the original '\|'.
        assert medoid in md


def test_render_pipe_in_medoid_escaped():
    df = env_agri_breakdown_df()
    md = render_breakdown_markdown(df)
    body_rows = [ln for ln in md.splitlines() if ln.startswith("| ") and "base_key" not in ln and "---" not in ln]
    assert body_rows, "expected at least one body row"
    for line in body_rows:
        # The breakdown has 5 columns (base_key, cluster_id, medoid,
        # cluster_size, total_count_all). A well-formed body row has
        # 6 top-level pipes (1 leading + 4 separators + 1 trailing),
        # giving 7 cells when split on un-escaped pipes.
        cleaned = line.replace("\\|", "\x00")
        cells = [c.strip() for c in cleaned.split("|")]
        assert len(cells) == 7, f"row not 5 columns: {line!r}"
