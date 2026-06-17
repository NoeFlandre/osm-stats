"""One-off script: build and write the TF-IDF vs Embeddings comparison artifact.

Reads the four pipeline artifacts (two per pipeline), joins them via
the ``src.core.features.embedding.comparison`` module, and writes a
single Markdown report to ``output/filter_first/comparison/pipeline_comparison.md``.

This is the **filter-first** variant. The parallel **standardize-first**
variant lives in ``scripts/compare_pipelines_standardize_first.py`` and
writes under ``output/standardize_first/comparison/``.

Run with:
    .venv/bin/python -m scripts.compare_pipelines

This is a runnable script, not a library function. The underlying
comparison module has its own test suite (``tests/core/features/embedding
/test_comparison.py``); this script is just glue.
"""
from pathlib import Path

from src.core.features.embedding.comparison import (
    build_comparison,
    build_env_agri_breakdown_from_medoids,
    build_env_agri_comparison,
    load_profile_from_markdown,
    render_comparison_markdown,
)


TFIDF_PROFILE = Path("output/filter_first/tfidf/cluster_profile.md")
TFIDF_MEDOIDS = Path("output/filter_first/tfidf/cluster_medoids.csv")
EMBED_PROFILE = Path("output/filter_first/embeddings/cluster_profile_embeddings.md")
EMBED_MEDOIDS = Path("output/filter_first/embeddings/cluster_medoids_embeddings.csv")
COMPARISON_OUT = Path("output/filter_first/comparison/pipeline_comparison.md")


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
    COMPARISON_OUT.write_text(md)

    print(f"comparison rows: {len(comparison_df)}")
    print(f"env/agri rows:   {len(env_agri_df)}")
    print()
    print("top 10 base keys by combined volume (cluster_count_delta, total_count_all_delta):")
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
