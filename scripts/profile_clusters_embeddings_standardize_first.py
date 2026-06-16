"""End-to-end embeddings pipeline on the **standardize-first** cache.

Shape-equivalent to ``scripts/profile_clusters_embeddings.py`` (the
filter-first embeddings pipeline), but reads the standardize-first
cache and writes to ``output/standardize_first/embeddings/``. The
two embeddings outputs are first-class and never overwrite each
other.

Run with:
    .venv/bin/python -m scripts.profile_clusters_embeddings_standardize_first
"""
import time
from pathlib import Path

import pandas as pd

from src.core.features.embedding.runner import run


CACHE = "/Volumes/Seagate M3/tag_features_standardize_first.sqlite"
OUTPUT_DIR = Path("output/standardize_first/embeddings")
MIN_COUNT = 500


def main() -> None:
    wall_start = time.time()
    banner = "=" * 60
    print(banner)
    print("embedding pipeline (standardize-first cache)")
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

    # Top-20 of the profile, for human inspection.
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
