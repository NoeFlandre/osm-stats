"""Write the env/agri per-cluster breakdown to output/filter_first/tfidf/env_agri_breakdown.md.

The breakdown is sourced from the TF-IDF per-cluster medoid file
(persisted by ``scripts/profile_clusters.py``).

Run with:
    .venv/bin/python -m scripts.write_env_agri_breakdown
"""
from pathlib import Path

from src.core.features.breakdown import write_breakdown_artifact

OUTPUT = Path("output/filter_first/tfidf/env_agri_breakdown.md")


def main() -> None:
    path = write_breakdown_artifact(OUTPUT)
    print(f"wrote: {path}")


if __name__ == "__main__":
    main()
