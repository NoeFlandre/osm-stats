"""Entry point for the OSM taginfo analysis pipeline.

Run with::

    python -m src.main

Override the database location with the ``OSM_DB_PATH`` environment variable.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DEFAULT_OUTPUT_DIR, resolve_db_path
from src.core.audit import OSMTagAuditor
from src.core.database import OSMDatabase
from src.core.queries import QueryBuilder
from src.io.exporter import DataExporter


def build_summary(db: OSMDatabase) -> pd.DataFrame:
    """Aggregate the global key/tag totals into a single tidy DataFrame."""
    df_key_agg = db.execute_query(QueryBuilder.GLOBAL_KEY_AGGREGATES)
    df_tag_agg = db.execute_query(QueryBuilder.GLOBAL_TAG_AGGREGATES)
    return pd.DataFrame(
        {
            "metric": [
                "total_distinct_keys",
                "total_key_occurrences",
                "total_distinct_tags",
                "total_tag_occurrences",
            ],
            "value": [
                df_key_agg["total_distinct_keys"].iloc[0],
                df_key_agg["total_key_occurrences"].iloc[0],
                df_tag_agg["total_distinct_tags"].iloc[0],
                df_tag_agg["total_tag_occurrences"].iloc[0],
            ],
        }
    )


def run(db: OSMDatabase, output_dir: Path) -> pd.DataFrame:
    """Execute the analysis pipeline against *db* and write results to *output_dir*.

    Returns the building-audit DataFrame for inspection.
    """
    exporter = DataExporter(db, output_dir)

    summary = build_summary(db)
    summary.to_csv(output_dir / "global_summary.csv", index=False)

    auditor = OSMTagAuditor(db)
    df_building_audit = auditor.top_values("building", limit=50)
    df_building_audit.to_csv(output_dir / "audit_building.csv", index=False)
    return df_building_audit


def main(output_dir: Path = DEFAULT_OUTPUT_DIR) -> int:
    print("Initializing OSM Database analysis...")

    db_path = resolve_db_path()
    try:
        db = OSMDatabase(str(db_path))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    print("Auditing 'building' key...")
    df_building_audit = run(db, output_dir)

    print("\n--- Top 20 Building Tags ---")
    print(df_building_audit.head(20).to_string(index=False))
    print("----------------------------\n")

    print("Analysis complete! Results saved to the 'output' directory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
