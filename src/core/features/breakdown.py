"""Per-cluster breakdown for the env/agri whitelist.

The cluster profile (``output/cluster_profile.md``) emits one row per
base key, with the top-N representative medoids joined with ``"; "``.
The breakdown expands those representative medoids into a flat
per-cluster view, restricted to the env/agri whitelist, and renders it
as a Markdown table with one row per cluster.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.core.features.env_agri_whitelist import ENVI_AGRI_BASE_KEYS
from src.core.features.render import _escape_cell  # reuse the escape helper


_PROFILE_PATH = Path("output/cluster_profile.md")


def _parse_profile() -> list[dict]:
    md = _PROFILE_PATH.read_text()
    rows = []
    for line in md.splitlines():
        if not line.startswith("| ") or "base_key" in line or line.startswith("| ---"):
            continue
        # Split only on un-escaped pipes: replace escaped pipes with a
        # placeholder, split, then restore.
        placeholder = "\x00PIPE\x00"
        cleaned = line.replace("\\|", placeholder)
        cells = [c.strip() for c in cleaned.split("|")]
        cells = [c.replace(placeholder, "\\|") for c in cells]
        if len(cells) < 5:
            continue
        bk = cells[1]
        if not bk or bk == "---":
            continue
        rows.append(
            {
                "base_key": bk,
                "cluster_count": int(cells[2].replace(",", "")),
                "total_count_all": int(cells[3].replace(",", "")),
                "medoids": [m for m in cells[4].split("; ") if m],
            }
        )
    return rows


def env_agri_breakdown_df() -> pd.DataFrame:
    """One row per representative cluster medoid, restricted to the
    env/agri whitelist. Each row carries the medoid, its synthetic
    cluster_id (sequential 0..N-1 within the base key), and the
    total_count_all of the parent base key (so the user can see the
    family-level volume even though per-cluster counts aren't stored
    on disk).
    """
    rows: list[dict] = []
    cid = 0
    for entry in _parse_profile():
        if entry["base_key"] not in ENVI_AGRI_BASE_KEYS:
            continue
        for medoid in entry["medoids"]:
            rows.append(
                {
                    "base_key": entry["base_key"],
                    "cluster_id": cid,
                    "medoid": medoid,
                    "cluster_size": entry["cluster_count"],
                    "total_count_all": entry["total_count_all"],
                }
            )
            cid += 1
    return pd.DataFrame(
        rows,
        columns=["base_key", "cluster_id", "medoid", "cluster_size", "total_count_all"],
    )


def render_breakdown_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    cols = ["base_key", "cluster_id", "medoid", "cluster_size", "total_count_all"]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = [
            _escape_cell(str(row["base_key"])),
            _escape_cell(str(int(row["cluster_id"]))),
            _escape_cell(str(row["medoid"])),
            _escape_cell(f"{int(row['cluster_size']):,}"),
            _escape_cell(f"{int(row['total_count_all']):,}"),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
