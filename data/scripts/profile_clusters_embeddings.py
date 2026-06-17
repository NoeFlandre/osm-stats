"""End-to-end: cache -> embeddings -> SVD -> HDBSCAN -> medoids -> profile -> markdown.

This is the embedding-based counterpart of
``scripts/profile_clusters.py``. It is shape-equivalent: the same
five stages, the same outputs, but driven by a static vector model
(``potion-base-8M``) instead of character n-gram TF-IDF.

This is the **filter-first** pipeline variant. The parallel
**standardize-first** variant lives in
``scripts/profile_clusters_embeddings_standardize_first.py`` and
writes under ``output/standardize_first/embeddings/``.

The script writes to *different* output paths than the TF-IDF
script so the two pipelines can be run back-to-back without
clobbering each other's artifacts:

    TF-IDF   :  output/filter_first/tfidf/cluster_profile.md
                output/filter_first/tfidf/cluster_medoids.csv
                output/filter_first/tfidf/env_agri_breakdown.md
                output/filter_first/tfidf/cluster_memberships.csv
    Embedding:  output/filter_first/embeddings/cluster_profile_embeddings.md
                output/filter_first/embeddings/cluster_medoids_embeddings.csv
                output/filter_first/embeddings/env_agri_breakdown_embeddings.md
                output/filter_first/embeddings/cluster_memberships_embeddings.csv

Run with:
    .venv/bin/python -m scripts.profile_clusters_embeddings
"""
import time
from pathlib import Path

import pandas as pd

from src.core.features.embedding.runner import run


CACHE = "/Volumes/Seagate M3/tag_features.sqlite"
OUTPUT_DIR = Path("output/filter_first/embeddings")
MIN_COUNT = 500


def main() -> None:
    wall_start = time.time()
    banner = "=" * 60
    print(banner)
    print("embedding pipeline: cache -> embed -> SVD -> HDBSCAN -> medoids")
    print(banner)
    print(f"cache: {CACHE}")
    print(f"output_dir: {OUTPUT_DIR}")
    print(f"min_count: {MIN_COUNT}")
    print()

    t0 = time.time()
    print("[1/3] running embedding pipeline...")
    summary = run(
        cache_path=CACHE,
        output_dir=OUTPUT_DIR,
        min_count=MIN_COUNT,
    )
    print(f"  done in {time.time() - t0:.1f}s")
    print()

    print("[2/3] summary")
    print(f"  n_tags:           {summary['n_tags']:,}")
    print(f"  n_clusters:       {summary['n_clusters']:,}")
    print(f"  n_noise:          {summary['n_noise']:,}")
    print(f"  noise_ratio:      {summary['noise_ratio']:.3f}")
    print(f"  medoid_count:     {summary['medoid_count']:,}")
    print(f"  embed_seconds:    {summary['embed_seconds']:.2f}")
    print(f"  svd_seconds:      {summary['svd_seconds']:.2f}")
    print(f"  hdbscan_seconds:  {summary['hdbscan_seconds']:.2f}")
    print()

    print("[3/3] artifacts")
    print(f"  profile:      {summary['profile_path']}")
    print(f"  medoids:      {summary['medoids_path']}")
    print(f"  breakdown:    {summary['breakdown_path']}")
    print(f"  memberships:  {summary['memberships_path']}")
    print()

    print(f"total wall time: {time.time() - wall_start:.1f}s")

    # Top-20 of the profile, for human inspection. We re-read the
    # persisted medoid CSV and the persisted profile Markdown to
    # avoid keeping the in-memory profile_df around.
    print()
    print("-" * 60)
    print("top 20 base keys (from medoids CSV)")
    print("-" * 60)
    medoids = pd.read_csv(summary["medoids_path"])
    real = medoids[medoids["cluster_id"] != -1].copy()
    if not real.empty:
        from src.core.features.base_key import parse_base_key

        real["base_key"] = real["medoid_feature"].map(parse_base_key)
        grouped = (
            real.groupby("base_key")
            .agg(
                cluster_count=("cluster_id", "count"),
                total_count_all=("total_count_all", "sum"),
            )
            .sort_values("total_count_all", ascending=False)
            .head(20)
            .reset_index()
        )
        print(grouped.to_string(index=False))
    else:
        print("(empty medoid file)")


if __name__ == "__main__":
    main()
