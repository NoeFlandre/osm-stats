"""Render the cluster-by-base-key profile as a Markdown table.

The output is a pipe-delimited Markdown table. To keep the table
parseable even though medoid strings contain their own pipe character
(``landuse|farmland``), each medoid is escaped by replacing ``|`` with
``\\|`` only inside the cell. The table is intended for human
inspection - the user can paste it into a blog post or a PR description
and read the dominant tag families at a glance.
"""
from __future__ import annotations

import pandas as pd


def _escape_cell(value: str) -> str:
    """Escape Markdown-significant characters in a single table cell."""
    return str(value).replace("|", "\\|")


def render_profile_markdown(profile: pd.DataFrame) -> str:
    """Render *profile* as a Markdown table sorted by total_count_all DESC."""
    if profile.empty:
        return ""

    cols = ["base_key", "cluster_count", "total_count_all", "representative_medoids"]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"

    lines = [header, sep]
    for _, row in profile.iterrows():
        cells = [
            _escape_cell(row["base_key"]),
            _escape_cell(f"{int(row['cluster_count']):,}"),
            _escape_cell(f"{int(row['total_count_all']):,}"),
            _escape_cell(row["representative_medoids"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
