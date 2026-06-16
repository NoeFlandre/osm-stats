"""End-to-end orchestrator for the embedding pipeline.

The runner is the *library* counterpart of
``scripts/profile_clusters_embeddings.py``. The script does the human
facing work (banners, time-stamping, printing the top-20 of the
profile); this module does the silent library work and returns a
summary dict the script can print.

Pipeline stages:

    read_cache -> embed -> SVD -> HDBSCAN -> medoids -> profile
                                            -> env/agri breakdown
                                            -> markdown artifacts

The TF-IDF pipeline (``scripts/profile_clusters.py``) is *not* called
from this module. The two pipelines are completely independent: they
read the same cache, but they write to *different* output paths so
neither can ever overwrite the other's artifacts.

Design notes
------------
* The default embedder is a :class:`model2vec.StaticModel` wrapped in
  a :class:`src.core.features.semantic_probe.SemanticProbe` so it
  exposes the ``.embed(tags)`` method that
  :func:`src.core.features.embedding.cluster_pipeline.run_embedding_pipeline`
  expects. Tests can pass a fake embedder via the ``embedder`` kwarg
  to avoid the network download.
* The default cache reader is :func:`src.core.storage.cache.read_cache_df`.
  Tests can pass a callable via the ``cache_reader`` kwarg.
* All timings use :func:`time.perf_counter` (monotonic, no wall-clock
  jumps). The script uses :func:`time.time` for human-readable output.
* The env/agri breakdown is built *locally* from the embedding medoid
  CSV. We do not import :func:`src.core.features.breakdown.env_agri_breakdown_df`
  because that function is hard-coded to read the TF-IDF medoid file.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Optional, Union

import pandas as pd

from src.core.features.base_key import parse_base_key
from src.core.features.cluster_memberships import save_cluster_memberships
from src.core.features.env_agri_whitelist import ENVI_AGRI_BASE_KEYS
from src.core.features.render import _escape_cell, render_profile_markdown


# Default output filenames. The TF-IDF pipeline uses ``cluster_profile.md``
# etc.; we deliberately use a different suffix to make it impossible for
# the embedding runner to clobber the TF-IDF outputs.
PROFILE_FILENAME = "cluster_profile_embeddings.md"
MEDOIDS_FILENAME = "cluster_medoids_embeddings.csv"
BREAKDOWN_FILENAME = "env_agri_breakdown_embeddings.md"
MEMBERSHIPS_FILENAME = "cluster_memberships_embeddings.csv"

DEFAULT_MODEL_NAME = "minishlab/potion-base-8M"
DEFAULT_BATCH_SIZE = 4096
DEFAULT_N_COMPONENTS = 50
DEFAULT_MIN_CLUSTER_SIZE = 5
DEFAULT_MIN_SAMPLES = 2
DEFAULT_MIN_COUNT = 500


# ---------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------

# A cache reader returns a DataFrame. Signature mirrors
# :func:`src.core.storage.cache.read_cache_df`.
CacheReader = Callable[..., pd.DataFrame]

# An embedder exposes ``.embed(tags) -> np.ndarray`` (any duck-typed
# object with that method works).
Embedder = Any


# ---------------------------------------------------------------------
# Helpers (small, pure, easy to test)
# ---------------------------------------------------------------------


def _build_env_agri_breakdown_df(medoids_path: Path) -> pd.DataFrame:
    """Build the env/agri breakdown DataFrame from an embedding medoid CSV.

    Reads *medoids_path* (the embedding medoid CSV), parses the base
    key out of each ``medoid_feature``, filters to the
    :data:`ENVI_AGRI_BASE_KEYS` whitelist, and returns a DataFrame
    with columns ``base_key, cluster_id, medoid, cluster_size,
    total_count_all`` sorted by ``total_count_all`` descending.

    The schema matches the one produced by
    :func:`src.core.features.breakdown.env_agri_breakdown_df` so the
    two pipelines' breakdowns are interchangeable downstream.
    """
    if not medoids_path.exists():
        raise FileNotFoundError(
            f"missing {medoids_path}: re-run the embedding pipeline "
            f"to produce the per-cluster medoid file"
        )
    src = pd.read_csv(medoids_path)
    src = src.copy()
    src["base_key"] = src["medoid_feature"].map(parse_base_key)
    src = src[src["base_key"].isin(ENVI_AGRI_BASE_KEYS)]
    src = src.rename(columns={"medoid_feature": "medoid"})
    src = src[
        ["base_key", "cluster_id", "medoid", "cluster_size", "total_count_all"]
    ]
    src = src.sort_values("total_count_all", ascending=False).reset_index(drop=True)
    return src


def _render_embedding_breakdown_markdown(df: pd.DataFrame) -> str:
    """Render the embedding env/agri breakdown as a Markdown table.

    Local re-implementation of
    :func:`src.core.features.breakdown.render_breakdown_markdown`. The
    TF-IDF version reads from a hard-coded path, which is the wrong
    input for the embedding pipeline. The two implementations are
    otherwise identical: same columns, same escape rules.
    """
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


def _build_default_embedder(model_name: str) -> Embedder:
    """Lazily build the default embedder (real ``potion-base-8M``).

    The default embedder is a
    :class:`src.core.features.semantic_probe.SemanticProbe` wrapping
    a :class:`model2vec.StaticModel`. The probe exposes the
    ``.embed(tags)`` method that
    :func:`run_embedding_pipeline` expects, while delegating the
    actual encoding to the static model's native batched encoder.
    """
    from model2vec import StaticModel

    from src.core.features.semantic_probe import SemanticProbe

    return SemanticProbe(embedder=StaticModel.from_pretrained(model_name))


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def run(
    *,
    cache_path: Union[str, Path],
    output_dir: Union[str, Path],
    min_count: int = DEFAULT_MIN_COUNT,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_components: int = DEFAULT_N_COMPONENTS,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    limit: Optional[int] = None,
    embedder: Optional[Embedder] = None,
    cache_reader: Optional[CacheReader] = None,
) -> dict:
    """Run the embedding pipeline end-to-end. Returns a summary dict.

    The function is silent: no prints, no banners, no logging. The
    caller is responsible for human-facing output. All timings use
    :func:`time.perf_counter` and are reported in seconds as floats.

    Parameters
    ----------
    cache_path:
        Path to the SQLite cache file. The cache must follow the
        ``tag_features`` schema defined in
        :mod:`src.core.storage.cache`.
    output_dir:
        Directory where the three artifacts are written. The
        directory is created if it does not exist. Existing
        files inside the directory that are *not* the three
        embedding outputs are left alone.
    min_count:
        Forwarded to the cache reader. Rows with ``count_all``
        below this threshold are dropped before embedding.
    model_name:
        Hugging Face model id used when *embedder* is ``None``.
        Ignored when *embedder* is supplied.
    batch_size:
        Forwarded to the embedder. 4096 is a comfortable default
        that fits easily in CPU RAM for 384-dim float32 vectors.
    n_components:
        Forwarded to the SVD reduction step (50 by default). Larger
        values are clamped by :class:`sklearn.decomposition.TruncatedSVD`.
    min_cluster_size:
        Forwarded to HDBSCAN (5 by default).
    min_samples:
        Forwarded to HDBSCAN (2 by default).
    limit:
        Optional cap on the number of cache rows to read. Useful
        for the slow integration test that runs on a 500-row slice.
    embedder:
        Optional pre-built embedder. Any object with an
        ``embed(tags) -> np.ndarray`` method works. When ``None``,
        a real :class:`model2vec.StaticModel` for *model_name* is
        loaded lazily on first use.
    cache_reader:
        Optional pre-built cache reader. When ``None``, the default
        :func:`src.core.storage.cache.read_cache_df` is used.

    Returns
    -------
    dict
        Summary with keys ``n_tags``, ``embed_seconds``,
        ``svd_seconds``, ``hdbscan_seconds``, ``medoid_count``,
        ``n_clusters``, ``n_noise``, ``noise_ratio``,
        ``profile_path``, ``medoids_path``, ``breakdown_path``,
        ``memberships_path``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read the cache.
    if cache_reader is None:
        from src.core.storage.cache import read_cache_df

        cache_reader = read_cache_df
    df = cache_reader(cache_path, min_count=min_count, limit=limit)
    n_tags = int(len(df))

    # 2. Build the embedder lazily.
    if embedder is None:
        embedder = _build_default_embedder(model_name)

    # 3. Run the embed -> SVD -> HDBSCAN -> medoids -> profile stages.
    # We time the three sub-stages independently and feed the
    # intermediate results into the next stage. This keeps the
    # sub-timings honest (no double-counting from a wrapper
    # function).
    tags = df["feature"].tolist()
    counts = df["count_all"].tolist()

    # We import lazily to keep the public API path import-cheap for
    # callers that just want the summary dict shape.
    from src.core.features.cluster import cluster_tags
    from src.core.features.medoids import compute_cluster_medoids
    from src.core.features.profile import profile_clusters_by_base_key
    from src.core.features.reduce import reduce_dimensions
    from src.core.features.semantic_probe import SemanticProbe  # noqa: F401  (kept for type docs)

    # The default embedder we build already exposes ``.embed``. For
    # ad-hoc embedders (e.g. raw ``StaticModel``) we transparently
    # wrap them in a ``SemanticProbe`` so the contract holds.
    if not hasattr(embedder, "embed"):
        from src.core.features.semantic_probe import SemanticProbe as _SP

        embedder = _SP(embedder=embedder)

    if n_tags == 0:
        # Short-circuit: no tags to embed. The downstream stages
        # would crash; we return an empty result instead.
        empty_md = ""
        empty_csv = "cluster_id,medoid_feature,cluster_size,total_count_all\n"
        empty_memberships = "cluster_id,base_key,key,value,feature,count_all\n"
        profile_path = output_dir / PROFILE_FILENAME
        medoids_path = output_dir / MEDOIDS_FILENAME
        breakdown_path = output_dir / BREAKDOWN_FILENAME
        memberships_path = output_dir / MEMBERSHIPS_FILENAME
        profile_path.write_text(empty_md)
        medoids_path.write_text(empty_csv)
        breakdown_path.write_text(empty_md)
        memberships_path.write_text(empty_memberships)
        return {
            "n_tags": 0,
            "embed_seconds": 0.0,
            "svd_seconds": 0.0,
            "hdbscan_seconds": 0.0,
            "medoid_count": 0,
            "n_clusters": 0,
            "n_noise": 0,
            "noise_ratio": 0.0,
            "profile_path": profile_path,
            "medoids_path": medoids_path,
            "breakdown_path": breakdown_path,
            "memberships_path": memberships_path,
        }

    # 3a. Embed
    t_embed_start = time.perf_counter()
    vecs = embedder.embed(tags)
    embed_seconds = time.perf_counter() - t_embed_start

    # 3b. SVD
    t_svd_start = time.perf_counter()
    dense = reduce_dimensions(vecs, n_components=n_components)
    svd_seconds = time.perf_counter() - t_svd_start

    # 3c. HDBSCAN
    t_hdbscan_start = time.perf_counter()
    labels = cluster_tags(
        dense, min_cluster_size=min_cluster_size, min_samples=min_samples
    )
    hdbscan_seconds = time.perf_counter() - t_hdbscan_start

    # 3d. Medoids + profile (these are deterministic and cheap;
    # no separate timing is needed for the summary).
    medoids_df = compute_cluster_medoids(
        features=tags, dense=dense, labels=labels, counts=counts
    )
    profile_df = profile_clusters_by_base_key(medoids_df, top_n=5)

    distinct = {int(x) for x in labels.tolist()}
    n_clusters = len(distinct - {-1})
    n_noise = int((labels == -1).sum())
    medoid_count = int(len(medoids_df))
    noise_ratio = float(n_noise / n_tags) if n_tags > 0 else 0.0

    # 4. Write the four artifacts.
    profile_path = output_dir / PROFILE_FILENAME
    medoids_path = output_dir / MEDOIDS_FILENAME
    breakdown_path = output_dir / BREAKDOWN_FILENAME
    memberships_path = output_dir / MEMBERSHIPS_FILENAME

    profile_md = render_profile_markdown(profile_df)
    profile_path.write_text(profile_md + "\n" if profile_md else "")

    medoids_df.to_csv(medoids_path, index=False)

    breakdown_df = _build_env_agri_breakdown_df(medoids_path)
    breakdown_md = _render_embedding_breakdown_markdown(breakdown_df)
    breakdown_path.write_text(breakdown_md + "\n" if breakdown_md else "")

    # Per-tag membership CSV: one row per cache tag with its cluster
    # assignment. This is the raw, unfiltered output of HDBSCAN: the
    # env/agri filter is never applied here. The user reads this file
    # to decide which clusters to keep.
    save_cluster_memberships(labels, df, memberships_path)

    return {
        "n_tags": n_tags,
        "embed_seconds": float(embed_seconds),
        "svd_seconds": float(svd_seconds),
        "hdbscan_seconds": float(hdbscan_seconds),
        "medoid_count": medoid_count,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_ratio": noise_ratio,
        "profile_path": profile_path,
        "medoids_path": medoids_path,
        "breakdown_path": breakdown_path,
        "memberships_path": memberships_path,
    }
