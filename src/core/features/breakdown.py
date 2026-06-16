"""Per-cluster breakdown for the env/agri whitelist.

The cluster step persists a per-cluster medoid file
(``output/filter_first/tfidf/cluster_medoids.csv``) with one row per
real cluster. The breakdown filters that file to the whitelisted
env/agri base keys and renders it as a Markdown table with one row
per cluster.

Each row carries the real per-cluster ``cluster_id`` and the real
per-cluster ``total_count_all`` (the sum of ``count_all`` across the
cluster's members). This is not a recomputation: the breakdown is a
filter on the persisted medoid file.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core.features.base_key import parse_base_key
from src.core.features.env_agri_whitelist import ENVI_AGRI_BASE_KEYS
from src.core.features.render import _escape_cell  # reuse the escape helper


MEDOIDS_PATH = Path("output/filter_first/tfidf/cluster_medoids.csv")


def _load_medoids() -> pd.DataFrame:
    """Load the persisted per-cluster medoid file.

    The medoid file has columns ``cluster_id``, ``medoid_feature``,
    ``cluster_size``, ``total_count_all``. The noise bucket (label -1)
    is included; its ``medoid_feature`` is the top noise feature.
    """
    if not MEDOIDS_PATH.exists():
        raise FileNotFoundError(
            f"missing {MEDOIDS_PATH}: re-run scripts/profile_clusters.py "
            f"to produce the per-cluster medoid file"
        )
    df = pd.read_csv(MEDOIDS_PATH)
    return df


def env_agri_breakdown_df() -> pd.DataFrame:
    """One row per real cluster whose base key is in the env/agri whitelist.

    Columns: ``base_key``, ``cluster_id``, ``medoid``, ``cluster_size``,
    ``total_count_all``. Sorted by ``total_count_all`` descending.
    """
    src = _load_medoids()
    src = src.copy()
    src["base_key"] = src["medoid_feature"].map(parse_base_key)
    src = src[src["base_key"].isin(ENVI_AGRI_BASE_KEYS)]
    src = src.rename(columns={"medoid_feature": "medoid"})
    src = src[["base_key", "cluster_id", "medoid", "cluster_size", "total_count_all"]]
    src = src.sort_values("total_count_all", ascending=False).reset_index(drop=True)
    return src


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


def write_breakdown_artifact(output_path: Path) -> Path:
    """Write the env/agri breakdown as a Markdown artifact on disk.

    Returns the path that was written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = env_agri_breakdown_df()
    output_path.write_text(render_breakdown_markdown(df) + "\n")
    return output_path
