"""TF-IDF vs Embeddings comparison on the **standardize-first** cache.

Reads the four pipeline artifacts (two per pipeline) from
``output/standardize_first/``, joins them via
``src.core.features.embedding.comparison``, and writes a single
Markdown report to ``output/standardize_first/comparison/pipeline_comparison.md``.

Run with:
    .venv/bin/python -m scripts.compare_pipelines_standardize_first
"""
from pathlib import Path

from src.core.features.embedding.comparison import (
    build_comparison,
    build_env_agri_breakdown_from_medoids,
    build_env_agri_comparison,
    load_profile_from_markdown,
    render_comparison_markdown,
)


TFIDF_PROFILE = Path("output/standardize_first/tfidf/cluster_profile.md")
TFIDF_MEDOIDS = Path("output/standardize_first/tfidf/cluster_medoids.csv")
EMBED_PROFILE = Path("output/standardize_first/embeddings/cluster_profile_embeddings.md")
EMBED_MEDOIDS = Path("output/standardize_first/embeddings/cluster_medoids_embeddings.csv")
COMPARISON_OUT = Path("output/standardize_first/comparison/pipeline_comparison.md")


def main() -> None:
    tfidf_profile_df = load_profile_from_markdown(TFIDF_PROFILE)
    embedding_profile_df = load_profile_from_markdown(EMBED_PROFILE)
    comparison_df = build_comparison(tfidf_profile_df, embedding_profile_df)

    tfidf_breakdown_df = build_env_agri_breakdown_from_medoids(TFIDF_MEDOIDS)
    embedding_breakdown_df = build_env_agri_breakdown_from_medoids(EMBED_MEDOIDS)
    env_agri_df = build_env_agri_comparison(
        tfidf_breakdown_df, embedding_breakdown_df
    )

    md = render_comparison_markdown(comparison_df, env_agri_df)
    COMPARISON_OUT.parent.mkdir(parents=True, exist_ok=True)
    COMPARISON_OUT.write_text(md)

    print(f"comparison rows: {len(comparison_df)}")
    print(f"env/agri rows:   {len(env_agri_df)}")
    print()
    print("top 10 base keys by combined volume:")
    print(
        comparison_df.head(10)[
            [
                "base_key",
                "tfidf_cluster_count",
                "embedding_cluster_count",
                "cluster_count_delta",
                "tfidf_total_count_all",
                "embedding_total_count_all",
                "total_count_all_delta",
            ]
        ].to_string(index=False)
    )
    print()
    print("env/agri section:")
    print(env_agri_df.to_string(index=False))
    print()
    print(f"wrote: {COMPARISON_OUT}")


if __name__ == "__main__":
    main()
