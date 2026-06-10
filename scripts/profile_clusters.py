"""End-to-end: cache -> TF-IDF -> SVD -> HDBSCAN -> medoids -> profile -> markdown.

Run with:
    .venv/bin/python -m scripts.profile_clusters
"""
import time
from pathlib import Path

from src.core.features.cluster import cluster_tags
from src.core.features.medoids import compute_cluster_medoids
from src.core.features.profile import profile_clusters_by_base_key
from src.core.features.reduce import reduce_dimensions
from src.core.features.render import render_profile_markdown
from src.core.features.tfidf import build_char_tfidf_matrix
from src.core.storage.cache import read_cache_df

CACHE = "/Volumes/Seagate M3/tag_features.sqlite"
OUTPUT = Path("output/cluster_profile.md")
MIN_COUNT = 500


def main() -> None:
    t0 = time.time()
    df = read_cache_df(CACHE, min_count=MIN_COUNT)
    print(f"load: {time.time()-t0:.1f}s  rows: {len(df):,}")

    t0 = time.time()
    matrix, _ = build_char_tfidf_matrix(df["feature"].tolist())
    print(f"tfidf: {time.time()-t0:.1f}s  shape: {matrix.shape}")

    t0 = time.time()
    dense = reduce_dimensions(matrix, n_components=50)
    print(f"svd: {time.time()-t0:.1f}s  shape: {dense.shape}")

    t0 = time.time()
    labels = cluster_tags(dense, min_cluster_size=5, min_samples=2)
    n_clusters = len(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    print(
        f"hdbscan: {time.time()-t0:.1f}s  clusters: {n_clusters:,}  noise: {n_noise:,}"
    )

    t0 = time.time()
    medoids = compute_cluster_medoids(
        features=df["feature"].tolist(),
        dense=dense,
        labels=labels,
        counts=df["count_all"].tolist(),
    )
    print(f"medoids: {time.time()-t0:.1f}s  rows: {len(medoids):,}")

    t0 = time.time()
    profile = profile_clusters_by_base_key(medoids, top_n=5)
    print(f"profile: {time.time()-t0:.1f}s  base_keys: {len(profile)}")

    md = render_profile_markdown(profile)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(md + "\n")
    print(f"wrote: {OUTPUT}")

    print()
    print(profile.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
