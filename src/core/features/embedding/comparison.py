"""Side-by-side comparison of the TF-IDF and Embeddings pipelines.

The two pipelines each persist two artifacts on disk:

* ``output/filter_first/tfidf/cluster_profile.md`` (TF-IDF) or
  ``output/filter_first/embeddings/cluster_profile_embeddings.md``
  (Embeddings) - a Markdown table with one row per base-key family
  and the columns ``base_key``, ``cluster_count``,
  ``total_count_all``, ``representative_medoids``.
* ``output/filter_first/tfidf/cluster_medoids.csv`` (TF-IDF) or
  ``output/filter_first/embeddings/cluster_medoids_embeddings.csv``
  (Embeddings) - one row per real cluster (and a noise bucket) with
  the columns ``cluster_id``, ``medoid_feature``, ``cluster_size``,
  ``total_count_all``.

This module reads those artifacts back, joins them on ``base_key``,
computes deltas, and renders a single Markdown report. It is a *pure*
module: it does not call into the TF-IDF breakdown module
(``src.core.features.breakdown``), because that module is hard-coded
to read the TF-IDF medoid file. Decoupling the comparison from the
TF-IDF breakdown keeps the two pipelines symmetric and lets the
report be regenerated without re-running HDBSCAN.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core.features.base_key import parse_base_key
from src.core.features.env_agri_whitelist import ENVI_AGRI_BASE_KEYS
from src.core.features.render import _escape_cell


# Placeholder used to make pipe-escaped cells safely splittable.
_PIPE_PLACEHOLDER = "\x00P\x00"


# ---------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------


def load_profile_from_markdown(path: Path) -> pd.DataFrame:
    """Parse a pipeline's Markdown profile into a DataFrame.

    Returns one row per base-key family with columns
    ``base_key``, ``cluster_count``, ``total_count_all``,
    ``representative_medoids``. Escaped pipes (``\\|``) inside the
    medoid cells are unescaped back to ``|``; comma-separated numbers
    (e.g. ``4,790``) are converted to ``int``.
    """
    text = Path(path).read_text()
    rows = []
    for line in text.splitlines():
        if not line.startswith("| "):
            continue
        if "base_key" in line or line.startswith("| ---"):
            continue

        # Replace escaped pipes with a placeholder so a naive split
        # on '|' does not break apart medoid cells like
        # 'landuse\|orchard'.
        masked = line.replace("\\|", _PIPE_PLACEHOLDER)
        cells = [c.strip() for c in masked.split("|")]
        # Drop the leading and trailing empty cells (the row starts
        # and ends with a '|').
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]
        if len(cells) < 4:
            continue

        base_key, cluster_count, total_count_all, *rest = cells
        medoid = _PIPE_PLACEHOLDER.join(rest)
        # Unescape the placeholder back to a literal '|'.
        medoid = medoid.replace(_PIPE_PLACEHOLDER, "|")

        rows.append(
            {
                "base_key": base_key,
                "cluster_count": int(cluster_count.replace(",", "")),
                "total_count_all": int(total_count_all.replace(",", "")),
                "representative_medoids": medoid,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "base_key",
            "cluster_count",
            "total_count_all",
            "representative_medoids",
        ],
    )


def load_medoids_from_csv(path: Path) -> pd.DataFrame:
    """Read a pipeline's per-cluster medoid CSV.

    The returned frame has the columns ``cluster_id``,
    ``medoid_feature``, ``cluster_size``, ``total_count_all``. Extra
    columns (e.g. ``top_features`` on the noise row) are preserved.
    The ``base_key`` column, if present in the file, is preserved
    too; the env/agri breakdown is built by re-deriving ``base_key``
    from ``medoid_feature`` so the comparison module does not depend
    on the TF-IDF pipeline's choice of whether to persist that
    column.
    """
    return pd.read_csv(Path(path))


# ---------------------------------------------------------------------
# Comparison builders
# ---------------------------------------------------------------------


def build_comparison(
    tfidf_profile: pd.DataFrame, embedding_profile: pd.DataFrame
) -> pd.DataFrame:
    """Join two base-key profiles and compute per-base-key deltas.

    Includes every base key seen in either profile (outer join).
    Sorted by absolute ``total_count_all_delta`` descending, then by
    ``base_key`` ascending for ties.
    """
    tfidf = tfidf_profile[
        ["base_key", "cluster_count", "total_count_all"]
    ].rename(
        columns={
            "cluster_count": "tfidf_cluster_count",
            "total_count_all": "tfidf_total_count_all",
        }
    )
    embed = embedding_profile[
        ["base_key", "cluster_count", "total_count_all"]
    ].rename(
        columns={
            "cluster_count": "embedding_cluster_count",
            "total_count_all": "embedding_total_count_all",
        }
    )

    merged = tfidf.merge(embed, on="base_key", how="outer")
    # NaN-fill the join misses so the deltas and pct_change are
    # well-defined.
    for col in (
        "tfidf_cluster_count",
        "embedding_cluster_count",
        "tfidf_total_count_all",
        "embedding_total_count_all",
    ):
        merged[col] = merged[col].fillna(0).astype(int)

    merged["cluster_count_delta"] = (
        merged["embedding_cluster_count"] - merged["tfidf_cluster_count"]
    ).astype(int)
    merged["total_count_all_delta"] = (
        merged["embedding_total_count_all"] - merged["tfidf_total_count_all"]
    ).astype(int)

    def _pct_change(row: pd.Series) -> float:
        tfidf_v = float(row["tfidf_total_count_all"])
        if tfidf_v == 0:
            return 0.0
        return round(
            float(row["total_count_all_delta"]) / tfidf_v * 100.0, 1
        )

    merged["total_count_all_pct_change"] = merged.apply(_pct_change, axis=1)

    merged["_abs_delta"] = merged["total_count_all_delta"].abs()
    merged = merged.sort_values(
        by=["_abs_delta", "base_key"],
        ascending=[False, True],
    ).drop(columns=["_abs_delta"]).reset_index(drop=True)

    return merged[
        [
            "base_key",
            "tfidf_cluster_count",
            "embedding_cluster_count",
            "cluster_count_delta",
            "tfidf_total_count_all",
            "embedding_total_count_all",
            "total_count_all_delta",
            "total_count_all_pct_change",
        ]
    ]


def build_env_agri_breakdown_from_medoids(
    medoids_csv_path: Path,
) -> pd.DataFrame:
    """Return the env/agri-filtered subset of a medoid CSV.

    Reads *medoids_csv_path* (a TF-IDF or Embeddings per-cluster
    medoid file), derives ``base_key`` from the ``medoid_feature``
    column via :func:`src.core.features.base_key.parse_base_key`,
    and returns the rows whose ``base_key`` is in
    :data:`ENVI_AGRI_BASE_KEYS`.

    Columns: ``base_key``, ``cluster_id``, ``medoid``,
    ``cluster_size``, ``total_count_all``. Sorted by
    ``total_count_all`` descending.
    """
    path = Path(medoids_csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}: re-run the pipeline to produce the "
            f"per-cluster medoid file"
        )
    src = pd.read_csv(path)
    src = src.copy()
    src["base_key"] = src["medoid_feature"].map(parse_base_key)
    src = src[src["base_key"].isin(ENVI_AGRI_BASE_KEYS)]
    src = src.rename(columns={"medoid_feature": "medoid"})
    src = src[
        ["base_key", "cluster_id", "medoid", "cluster_size", "total_count_all"]
    ]
    src = src.sort_values("total_count_all", ascending=False).reset_index(drop=True)
    return src


def build_env_agri_comparison(
    tfidf_breakdown: pd.DataFrame, embedding_breakdown: pd.DataFrame
) -> pd.DataFrame:
    """Build the per-base-key env/agri comparison.

    Aggregates ``tfidf_breakdown`` and ``embedding_breakdown`` (each
    one row per cluster) by ``base_key``:

    * ``tfidf_clusters`` / ``embedding_clusters``: count of distinct
      ``cluster_id`` values for the base key.
    * ``tfidf_occurrences`` / ``embedding_occurrences``: sum of
      ``total_count_all`` for the base key.
    * ``clusters_delta`` / ``occurrences_delta``: embedding minus
      tfidf.

    Outer-joined on ``base_key`` so base keys that appear in only one
    pipeline are still represented. Sorted by combined occurrence
    volume (``tfidf_occurrences + embedding_occurrences``)
    descending.
    """

    def _agg(df: pd.DataFrame, clusters_col: str, occ_col: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(
                columns=["base_key", clusters_col, occ_col]
            )
        return (
            df.groupby("base_key", as_index=False)
            .agg(
                **{
                    clusters_col: ("cluster_id", "nunique"),
                    occ_col: ("total_count_all", "sum"),
                }
            )
            .astype({clusters_col: int, occ_col: int})
        )

    tfidf_agg = _agg(tfidf_breakdown, "tfidf_clusters", "tfidf_occurrences")
    embed_agg = _agg(
        embedding_breakdown, "embedding_clusters", "embedding_occurrences"
    )

    merged = tfidf_agg.merge(embed_agg, on="base_key", how="outer")
    for col in (
        "tfidf_clusters",
        "embedding_clusters",
        "tfidf_occurrences",
        "embedding_occurrences",
    ):
        merged[col] = merged[col].fillna(0).astype(int)

    merged["clusters_delta"] = (
        merged["embedding_clusters"] - merged["tfidf_clusters"]
    ).astype(int)
    merged["occurrences_delta"] = (
        merged["embedding_occurrences"] - merged["tfidf_occurrences"]
    ).astype(int)

    merged["_sort_volume"] = (
        merged["tfidf_occurrences"] + merged["embedding_occurrences"]
    )
    merged = merged.sort_values(
        by=["_sort_volume", "base_key"],
        ascending=[False, True],
    ).drop(columns=["_sort_volume"]).reset_index(drop=True)

    return merged[
        [
            "base_key",
            "tfidf_clusters",
            "embedding_clusters",
            "clusters_delta",
            "tfidf_occurrences",
            "embedding_occurrences",
            "occurrences_delta",
        ]
    ]


# ---------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------


def _render_global_stats(comparison_df: pd.DataFrame) -> str:
    """Build the Global stats table from a comparison DataFrame.

    All needed totals are recoverable from the comparison frame:
    * n_base_keys = number of unique base keys in each pipeline.
      We approximate with the row count of the comparison and the
      count of non-zero entries in each side; but a cleaner source
      is the row count (one row per base key in either pipeline).
    * total clusters = sum of cluster counts.
    * total occurrences = sum of total_count_all values.
    """
    if comparison_df.empty:
        tfidf_n_keys = embed_n_keys = 0
        tfidf_n_clusters = embed_n_clusters = 0
        tfidf_n_occ = embed_n_occ = 0
    else:
        # n_base_keys: number of base keys that have a non-zero entry
        # in each pipeline. A base key only present in the embedding
        # profile contributes 0 to tfidf and 1 to embed; vice versa.
        tfidf_n_keys = int((comparison_df["tfidf_total_count_all"] > 0).sum())
        embed_n_keys = int((comparison_df["embedding_total_count_all"] > 0).sum())
        tfidf_n_clusters = int(comparison_df["tfidf_cluster_count"].sum())
        embed_n_clusters = int(comparison_df["embedding_cluster_count"].sum())
        tfidf_n_occ = int(comparison_df["tfidf_total_count_all"].sum())
        embed_n_occ = int(comparison_df["embedding_total_count_all"].sum())

    rows = [
        ("number of base keys", tfidf_n_keys, embed_n_keys),
        ("total clusters", tfidf_n_clusters, embed_n_clusters),
        (
            "total occurrences (top 20 base keys)",
            tfidf_n_occ,
            embed_n_occ,
        ),
    ]

    lines = [
        "| metric | tfidf | embeddings | delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, tfidf_v, embed_v in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_cell(label),
                    _escape_cell(f"{tfidf_v:,}"),
                    _escape_cell(f"{embed_v:,}"),
                    _escape_cell(f"{embed_v - tfidf_v:,}"),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _render_top_base_keys(comparison_df: pd.DataFrame, top_n: int = 20) -> str:
    header = "| " + " | ".join(
        [
            "base_key",
            "tfidf_clusters",
            "embedding_clusters",
            "clusters_delta",
            "tfidf_total",
            "embedding_total",
            "total_delta",
            "pct_change",
        ]
    ) + " |"
    sep = (
        "| "
        + " | ".join(
            [
                "---",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
            ]
        )
        + " |"
    )

    lines = [header, sep]
    if comparison_df.empty:
        return "\n".join(lines)
    top = comparison_df.head(top_n)
    for _, row in top.iterrows():
        cells = [
            _escape_cell(str(row["base_key"])),
            _escape_cell(f"{int(row['tfidf_cluster_count']):,}"),
            _escape_cell(f"{int(row['embedding_cluster_count']):,}"),
            _escape_cell(f"{int(row['cluster_count_delta']):,}"),
            _escape_cell(f"{int(row['tfidf_total_count_all']):,}"),
            _escape_cell(f"{int(row['embedding_total_count_all']):,}"),
            _escape_cell(f"{int(row['total_count_all_delta']):,}"),
            _escape_cell(f"{float(row['total_count_all_pct_change']):.1f}"),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _render_env_agri(env_agri_df: pd.DataFrame) -> str:
    cols = [
        "base_key",
        "tfidf_clusters",
        "embedding_clusters",
        "clusters_delta",
        "tfidf_occurrences",
        "embedding_occurrences",
        "occurrences_delta",
    ]
    header = "| " + " | ".join(cols) + " |"
    sep = (
        "| "
        + " | ".join(
            [
                "---",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
            ]
        )
        + " |"
    )
    lines = [header, sep]
    if env_agri_df.empty:
        return "\n".join(lines)
    for _, row in env_agri_df.iterrows():
        cells = [
            _escape_cell(str(row["base_key"])),
            _escape_cell(f"{int(row['tfidf_clusters']):,}"),
            _escape_cell(f"{int(row['embedding_clusters']):,}"),
            _escape_cell(f"{int(row['clusters_delta']):,}"),
            _escape_cell(f"{int(row['tfidf_occurrences']):,}"),
            _escape_cell(f"{int(row['embedding_occurrences']):,}"),
            _escape_cell(f"{int(row['occurrences_delta']):,}"),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_comparison_markdown(
    comparison_df: pd.DataFrame,
    env_agri_df: pd.DataFrame,
) -> str:
    """Render the comparison DataFrame and the env/agri comparison
    DataFrame as a single Markdown document.
    """
    parts: list[str] = [
        "# Pipeline Comparison: TF-IDF vs Embeddings",
        "",
        "## Global stats",
        "",
        _render_global_stats(comparison_df),
        "",
        "## Top 20 base keys by combined volume",
        "",
        _render_top_base_keys(comparison_df, top_n=20),
        "",
        "## Env/Agri whitelist comparison",
        "",
        _render_env_agri(env_agri_df),
        "",
    ]
    return "\n".join(parts)
