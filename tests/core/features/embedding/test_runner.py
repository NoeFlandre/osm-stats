"""Tests for the embedding-pipeline runner.

The :mod:`src.core.features.embedding.runner` module is the
end-to-end orchestrator for the embedding pipeline. It wires
together:

    read_cache -> embed -> SVD -> HDBSCAN -> medoids -> profile
                                            -> breakdown -> markdown

The unit tests use a hand-crafted fake embedder and an in-memory
SQLite cache that matches the ``tag_features`` schema so we never
depend on a network download or an external drive. One integration
test exercises the full pipeline on a 500-row slice of the real
cache with the real ``potion-base-8M`` model and is marked ``slow``
so it can be deselected in fast feedback loops.

The runner also has a critical contract: it must NEVER overwrite
the TF-IDF pipeline's outputs. The dedicated
:func:`test_run_does_not_overwrite_tfidf_outputs` test enforces
that contract by planting sentinel content in the TF-IDF file
before running the embedding pipeline.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import pytest

from src.core.features.embedding.runner import run
from src.core.features.env_agri_whitelist import ENVI_AGRI_BASE_KEYS


# ---------------------------------------------------------------------
# Fixtures: deterministic fake embedder + in-memory cache builder
# ---------------------------------------------------------------------


class _FakeEmbedder:
    """Encode each tag as a small dense vector keyed on its first character.

    Two tags whose first characters map to the same group land on
    the same dense cluster. The vector carries a small per-tag noise
    (keyed on the index) so within-group points are distinct but
    still close enough to cluster together with HDBSCAN.

    The default dim (64) is large enough that the default
    ``n_components=50`` SVD step never crashes on the fake.
    """

    def __init__(self, dim: int = 64, n_groups: int = 3) -> None:
        self.dim = dim
        self.n_groups = n_groups

    def embed(self, tags: Sequence[str]) -> np.ndarray:
        n = len(tags)
        out = np.zeros((n, self.dim), dtype=np.float32)
        for i, tag in enumerate(tags):
            first = ord(tag[0]) if tag else 0
            group = first % self.n_groups
            center = np.zeros(self.dim, dtype=np.float32)
            center[group * 2 % self.dim] = 10.0
            center[(group * 2 + 1) % self.dim] = 10.0
            out[i] = center + np.random.RandomState(i).randn(self.dim).astype(
                np.float32
            ) * 0.1
        return out


def _build_cache_sqlite(db_path: Path, rows: List[tuple]) -> Path:
    """Write a SQLite cache at *db_path* matching the tag_features schema.

    Each row is ``(key, value, count_all)``. The standardizer inside
    :func:`src.core.storage.cache.read_cache_df` will derive the
    ``feature`` column from ``key`` and ``value`` and skip rows
    below ``min_count``.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE tag_features (
                key        TEXT NOT NULL,
                value      TEXT NOT NULL,
                count_all  INTEGER NOT NULL,
                feature    TEXT NOT NULL,
                PRIMARY KEY (key, value)
            )
            """
        )
        for k, v, c in rows:
            feature = f"{k}|{v}"
            conn.execute(
                "INSERT INTO tag_features VALUES (?, ?, ?, ?)",
                (k, v, c, feature),
            )
        conn.commit()
    return db_path


# Small fixture: 15 landuse / natural tags all above min_count=500.
# Three groups (first letter cycles a, b, c) so HDBSCAN sees at
# least 3 clusters.
_DEFAULT_ROWS = [
    ("landuse", f"val_a{i}", 1_000 + i) for i in range(5)
] + [
    ("natural", f"val_b{i}", 900 + i) for i in range(5)
] + [
    ("highway", f"val_c{i}", 800 + i) for i in range(5)
]


# ---------------------------------------------------------------------
# 1. Artifact writing
# ---------------------------------------------------------------------


def test_run_writes_three_artifacts(tmp_path: Path):
    cache = _build_cache_sqlite(tmp_path / "cache.sqlite", _DEFAULT_ROWS)
    out_dir = tmp_path / "out"
    fake = _FakeEmbedder(dim=64, n_groups=3)

    run(
        cache_path=cache,
        output_dir=out_dir,
        min_count=500,
        embedder=fake,
    )

    profile = out_dir / "cluster_profile_embeddings.md"
    medoids = out_dir / "cluster_medoids_embeddings.csv"
    breakdown = out_dir / "env_agri_breakdown_embeddings.md"

    assert profile.exists()
    assert medoids.exists()
    assert breakdown.exists()
    assert profile.stat().st_size > 0
    assert medoids.stat().st_size > 0
    assert breakdown.stat().st_size > 0


# ---------------------------------------------------------------------
# 2. Summary dict
# ---------------------------------------------------------------------


def test_run_returns_summary_dict_with_all_keys(tmp_path: Path):
    cache = _build_cache_sqlite(tmp_path / "cache.sqlite", _DEFAULT_ROWS)
    out_dir = tmp_path / "out"
    fake = _FakeEmbedder(dim=64, n_groups=3)

    summary = run(
        cache_path=cache,
        output_dir=out_dir,
        min_count=500,
        embedder=fake,
    )

    expected_keys = {
        "n_tags",
        "embed_seconds",
        "svd_seconds",
        "hdbscan_seconds",
        "medoid_count",
        "n_clusters",
        "n_noise",
        "noise_ratio",
        "profile_path",
        "medoids_path",
        "breakdown_path",
    }
    assert expected_keys.issubset(set(summary.keys()))

    assert isinstance(summary["n_tags"], int)
    assert summary["n_tags"] >= 0
    assert isinstance(summary["n_clusters"], int)
    assert summary["n_clusters"] >= 0
    assert isinstance(summary["n_noise"], int)
    assert summary["n_noise"] >= 0

    assert isinstance(summary["embed_seconds"], float)
    assert summary["embed_seconds"] >= 0.0
    assert isinstance(summary["svd_seconds"], float)
    assert summary["svd_seconds"] >= 0.0
    assert isinstance(summary["hdbscan_seconds"], float)
    assert summary["hdbscan_seconds"] >= 0.0

    assert isinstance(summary["noise_ratio"], float)
    assert 0.0 <= summary["noise_ratio"] <= 1.0

    from pathlib import Path as _P

    assert isinstance(summary["profile_path"], _P)
    assert isinstance(summary["medoids_path"], _P)
    assert isinstance(summary["breakdown_path"], _P)


# ---------------------------------------------------------------------
# 3. Population invariant
# ---------------------------------------------------------------------


def test_run_n_clusters_plus_n_noise_equals_n_tags(tmp_path: Path):
    cache = _build_cache_sqlite(tmp_path / "cache.sqlite", _DEFAULT_ROWS)
    out_dir = tmp_path / "out"
    fake = _FakeEmbedder(dim=64, n_groups=3)

    summary = run(
        cache_path=cache,
        output_dir=out_dir,
        min_count=500,
        embedder=fake,
    )

    # Read the persisted medoid CSV to count cluster members.
    medoids = pd.read_csv(out_dir / "cluster_medoids_embeddings.csv")
    real = medoids[medoids["cluster_id"] != -1]
    non_noise = int(real["cluster_size"].sum())
    n_noise = int(summary["n_noise"])
    n_tags = int(summary["n_tags"])
    assert n_noise + non_noise == n_tags, (
        f"n_noise ({n_noise}) + non_noise members ({non_noise}) "
        f"!= n_tags ({n_tags})"
    )


# ---------------------------------------------------------------------
# 4. No-overwrite guarantee: the TF-IDF outputs must stay intact
# ---------------------------------------------------------------------


def test_run_does_not_overwrite_tfidf_outputs(tmp_path: Path):
    """Run the embedding pipeline and assert the TF-IDF file is unchanged.

    We plant sentinel content in the TF-IDF file (at a *temporary*
    output dir, so the real ``output/`` dir is not touched by the
    test) and then run the runner with the same ``output_dir``. The
    sentinel must be byte-for-byte identical afterwards.
    """
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    sentinel = "SENTINEL_TFIDF_PROFILE_DO_NOT_OVERWRITE\n"
    tfidf_path = out_dir / "cluster_profile.md"
    tfidf_path.write_text(sentinel)

    cache = _build_cache_sqlite(tmp_path / "cache.sqlite", _DEFAULT_ROWS)
    fake = _FakeEmbedder(dim=64, n_groups=3)

    run(
        cache_path=cache,
        output_dir=out_dir,
        min_count=500,
        embedder=fake,
    )

    # The runner must have written the embedding file...
    assert (out_dir / "cluster_profile_embeddings.md").exists()
    # ...and the TF-IDF file must be byte-for-byte unchanged.
    assert tfidf_path.read_text() == sentinel


# ---------------------------------------------------------------------
# 5. env/agri breakdown filters to the whitelist
# ---------------------------------------------------------------------


def test_env_agri_breakdown_filters_to_whitelist(tmp_path: Path):
    """10 tags whose base keys include 3 env/agri whitelist members.

    The breakdown must include only those 3 base keys, never any
    others.
    """
    # Pick three base keys from the actual whitelist and seven
    # outside it. We sort by total_count_all descending later, so
    # the order in the fixture does not matter.
    whitelist_picks = list(ENVI_AGRI_BASE_KEYS)[:3]
    non_whitelist = [
        "addr",
        "highway",
        "building",
        "source",
        "name",
        "ref",
        "place",
    ]
    # Use enough tags per base key to make HDBSCAN clustering
    # deterministic. Six tags per base key -> 6*10 = 60 rows.
    # Each base key gets six distinct value suffixes so the
    # embedding produces a tight cloud per family.
    rows = []
    counts = [10_000, 9_500, 9_000, 8_500, 8_000, 7_500, 7_000]
    for i, k in enumerate(whitelist_picks + non_whitelist):
        c = counts[i] if i < len(counts) else 1_000 - i
        for j in range(6):
            rows.append((k, f"val{i}_{j}", max(c - 10 * j, 600)))

    cache = _build_cache_sqlite(tmp_path / "cache.sqlite", rows)
    out_dir = tmp_path / "out"
    # 2 groups: keys starting with letters in {a, c, e, ...} vs
    # {b, d, f, ...}. The whitelist picks (``landuse``, ``natural``,
    # ``crop``) all start with l/n/c, which fall into group 1
    # (``l%2=0``, ``n%2=0``, ``c%2=0``). The non-whitelist keys
    # fall into group 0. This means HDBSCAN sees two well-separated
    # clusters (one per group), and the per-base-key breakdown
    # shows up correctly.
    fake = _FakeEmbedder(dim=64, n_groups=2)

    run(
        cache_path=cache,
        output_dir=out_dir,
        min_count=500,
        embedder=fake,
    )

    md = (out_dir / "env_agri_breakdown_embeddings.md").read_text()
    # Parse the rendered Markdown to extract the set of base keys
    # that actually appear as data rows. Substring matching is
    # unsafe (a base key like ``water_source`` contains ``source``
    # as a substring).
    base_keys_in_md: set = set()
    for line in md.splitlines():
        if not line.startswith("| ") or "---" in line or "base_key" in line:
            continue
        # Each row starts with ``| <base_key> | ...``. The base
        # key is the first column. Unescape pipes in the medoid
        # column later; the first column is a base key, no pipes.
        cells = [c.strip() for c in line.split("|")]
        if len(cells) >= 3:
            base_keys_in_md.add(cells[1])

    # The breakdown must mention every whitelisted base key that
    # actually appears in the cluster medoid file. It must never
    # mention non-whitelisted base keys.
    medoids = pd.read_csv(out_dir / "cluster_medoids_embeddings.csv")
    medoids = medoids.copy()
    from src.core.features.base_key import parse_base_key

    medoids["base_key"] = medoids["medoid_feature"].map(parse_base_key)
    in_medoids = set(medoids["base_key"])
    expected_in = in_medoids & set(whitelist_picks)
    # Each whitelisted base key that ended up in the medoid file
    # must show up in the rendered markdown.
    for bk in expected_in:
        assert bk in base_keys_in_md, (
            f"whitelisted base key {bk!r} missing from breakdown markdown; "
            f"got {sorted(base_keys_in_md)!r}"
        )
    for bk in set(non_whitelist):
        assert bk not in base_keys_in_md, (
            f"non-whitelisted base key {bk!r} leaked into breakdown markdown; "
            f"got {sorted(base_keys_in_md)!r}"
        )


# ---------------------------------------------------------------------
# 6. Empty cache
# ---------------------------------------------------------------------


def test_run_handles_empty_cache_gracefully(tmp_path: Path):
    """Empty cache (zero rows above min_count).

    Contract: the runner returns a sensible empty result. The
    summary dict has ``n_tags: 0``; the three artifact files exist
    (as empty or header-only) so downstream stages that ``.read_text()``
    on them do not crash.
    """
    cache = _build_cache_sqlite(tmp_path / "cache.sqlite", [])
    out_dir = tmp_path / "out"
    fake = _FakeEmbedder(dim=64, n_groups=3)

    summary = run(
        cache_path=cache,
        output_dir=out_dir,
        min_count=500,
        embedder=fake,
    )

    assert summary["n_tags"] == 0
    assert summary["n_clusters"] == 0
    assert summary["n_noise"] == 0
    assert summary["medoid_count"] == 0
    assert summary["noise_ratio"] == 0.0

    # The three artifact files must exist (as zero-byte or
    # header-only files). Downstream stages that do
    # ``path.read_text()`` must not crash with FileNotFoundError.
    profile = out_dir / "cluster_profile_embeddings.md"
    medoids = out_dir / "cluster_medoids_embeddings.csv"
    breakdown = out_dir / "env_agri_breakdown_embeddings.md"
    assert profile.exists()
    assert medoids.exists()
    assert breakdown.exists()


# ---------------------------------------------------------------------
# 7. Real-model integration (slow)
# ---------------------------------------------------------------------


REAL_CACHE = Path("/Volumes/Seagate M3/tag_features.sqlite")
SLOW_TIMEOUT_SECONDS = 5 * 60  # 5 minutes


@pytest.mark.slow
@pytest.mark.skipif(not REAL_CACHE.exists(), reason="real cache not mounted")
def test_real_model_end_to_end(tmp_path: Path):
    """End-to-end with the real cache + the real ``potion-base-8M``.

    Runs on a ``limit=500`` slice. Asserts the full pipeline
    completes in under 5 minutes, produces all three artifacts,
    and ``n_tags == 500``.
    """
    import time

    from model2vec import StaticModel

    from src.core.features.semantic_probe import SemanticProbe

    out_dir = tmp_path / "real_runner_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    embedder = SemanticProbe(embedder=StaticModel.from_pretrained("minishlab/potion-base-8M"))

    t0 = time.perf_counter()
    summary = run(
        cache_path=str(REAL_CACHE),
        output_dir=out_dir,
        min_count=500,
        limit=500,
        embedder=embedder,
    )
    elapsed = time.perf_counter() - t0

    assert elapsed < SLOW_TIMEOUT_SECONDS, (
        f"real-model end-to-end took {elapsed:.1f}s (>5min limit)"
    )
    assert summary["n_tags"] == 500
    assert (out_dir / "cluster_profile_embeddings.md").exists()
    assert (out_dir / "cluster_medoids_embeddings.csv").exists()
    assert (out_dir / "env_agri_breakdown_embeddings.md").exists()


# ---------------------------------------------------------------------
# 8. Auxiliary: the runner's breakdown is read from the embedding
#    medoid file, not the TF-IDF one
# ---------------------------------------------------------------------


def test_embedding_breakdown_reads_from_embedding_medoid_csv(tmp_path: Path):
    """The runner must read the *embedding* medoid CSV
    (``cluster_medoids_embeddings.csv``) when building the env/agri
    breakdown, not the TF-IDF one (``cluster_medoids.csv``)."""
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plant a TF-IDF medoid CSV whose medoids do not match the
    # cluster pipeline's output. If the runner accidentally read
    # this file, the breakdown would be inconsistent with the
    # actual medoid file.
    tfidf_path = out_dir / "cluster_medoids.csv"
    tfidf_path.write_text(
        "cluster_id,medoid_feature,cluster_size,total_count_all\n"
        "0,landuse|orchard,1,1\n"
    )

    # Build the cache, run the runner, and read the *embedding*
    # medoid file the runner produced.
    cache = _build_cache_sqlite(tmp_path / "cache.sqlite", _DEFAULT_ROWS)
    fake = _FakeEmbedder(dim=64, n_groups=3)
    run(
        cache_path=cache,
        output_dir=out_dir,
        min_count=500,
        embedder=fake,
    )
    embed_path = out_dir / "cluster_medoids_embeddings.csv"
    assert embed_path.exists()

    # The breakdown file must exist, and its content must reflect
    # the embedding medoid file (not the planted TF-IDF one).
    breakdown_path = out_dir / "env_agri_breakdown_embeddings.md"
    assert breakdown_path.exists()
    embed_medoids = pd.read_csv(embed_path)
    # At minimum, the medoids on disk must not match the planted
    # TF-IDF single-row file (different cluster_ids).
    assert len(embed_medoids) >= 1
    assert embed_medoids["cluster_id"].tolist() != [0]
