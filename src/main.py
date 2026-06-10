import os
import pandas as pd
from pathlib import Path
from src.database import OSMDatabase
from src.exporter import DataExporter
import src.queries as queries

DB_PATH = "/Volumes/Seagate M3/taginfo.sqlite"
OUTPUT_DIR = Path("output")

def main():
    print("Initializing OSM Database analysis...")
    
    try:
        db = OSMDatabase(DB_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    exporter = DataExporter(db, OUTPUT_DIR)

    # Global summary
    df_key_agg = db.execute_query(queries.GLOBAL_KEY_AGGREGATES_QUERY)
    df_tag_agg = db.execute_query(queries.GLOBAL_TAG_AGGREGATES_QUERY)
    
    summary_data = {
        "metric": ["total_distinct_keys", "total_key_occurrences", "total_distinct_tags", "total_tag_occurrences"],
        "value": [
            df_key_agg["total_distinct_keys"].iloc[0],
            df_key_agg["total_key_occurrences"].iloc[0],
            df_tag_agg["total_distinct_tags"].iloc[0],
            df_tag_agg["total_tag_occurrences"].iloc[0]
        ]
    }
    pd.DataFrame(summary_data).to_csv(OUTPUT_DIR / "global_summary.csv", index=False)
    
    # Audit 'building'
    print("Auditing 'building' key...")
    audit_query = queries.get_tag_values_query('building', limit=50)
    df_building_audit = exporter.export_query(audit_query, "audit_building.csv")
    
    print("\n--- Top 20 Building Tags ---")
    print(df_building_audit.head(20).to_string(index=False))
    print("----------------------------\n")

    print("Analysis complete! Results saved to the 'output' directory.")

if __name__ == "__main__":
    main()
