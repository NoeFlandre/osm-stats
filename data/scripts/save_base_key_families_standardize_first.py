"""Generate the per-pipeline base-key-family CSVs and XLSX files for
the **standardize-first** path.

The user can then open the two XLSX files in Excel, fill in the
``keep`` column for each base key (their env/agri verdict), and
later apply those verdicts to define the final kept-tag set.

Run with:
    .venv/bin/python -m scripts.save_base_key_families_standardize_first
"""
import os
from pathlib import Path

import pandas as pd

from src.core.features.base_key_families_io import (
    save_base_key_families_csv,
    save_base_key_families_xlsx,
)
from src.core.features.profile import profile_clusters_by_base_key


# Same source paths as the standardize-first cluster runs.
TARGETS = [
    (
        "output/standardize_first/tfidf/cluster_medoids.csv",
        "output/standardize_first/tfidf/base_key_families.csv",
        "output/standardize_first/tfidf/base_key_families.xlsx",
        "TF-IDF",
    ),
    (
        "output/standardize_first/embeddings/cluster_medoids_embeddings.csv",
        "output/standardize_first/embeddings/base_key_families.csv",
        "output/standardize_first/embeddings/base_key_families.xlsx",
        "Embeddings",
    ),
]


def main() -> None:
    for med_path, csv_path, xlsx_path, label in TARGETS:
        med = pd.read_csv(med_path)
        profile = profile_clusters_by_base_key(med, top_n=5)
        save_base_key_families_csv(profile, csv_path)
        save_base_key_families_xlsx(profile, xlsx_path, sheet_name=label)
        print(f"{label}: wrote {csv_path}  ({len(profile):,} base keys)")
        print(f"        wrote {xlsx_path}  ({os.path.getsize(xlsx_path):,} bytes)")


if __name__ == "__main__":
    main()
