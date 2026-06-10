import os
import pandas as pd
from pathlib import Path
from database import OSMDatabase
import queries

DB_PATH = "/Volumes/Seagate M3/taginfo.sqlite"
OUTPUT_DIR = Path("output")

def main():
    print("Initializing OSM Database analysis...")
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    try:
        db = OSMDatabase(DB_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print("Extracting global aggregates...")
    df_key_agg = db.execute_query(queries.GLOBAL_KEY_AGGREGATES_QUERY)
    df_tag_agg = db.execute_query(queries.GLOBAL_TAG_AGGREGATES_QUERY)
    
    summary_data = {
        "metric": [
            "total_distinct_keys", 
            "total_key_occurrences", 
            "total_distinct_tags", 
            "total_tag_occurrences"
        ],
        "value": [
            df_key_agg["total_distinct_keys"].iloc[0],
            df_key_agg["total_key_occurrences"].iloc[0],
            df_tag_agg["total_distinct_tags"].iloc[0],
            df_tag_agg["total_tag_occurrences"].iloc[0]
        ]
    }
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(OUTPUT_DIR / "global_summary.csv", index=False)
    
    print("\n--- Global Summary ---")
    print(df_summary.to_string(index=False))
    print("----------------------\n")

    # 1. Extract Metadata
    print("Extracting metadata...")
    df_meta = db.execute_query(queries.METADATA_QUERY)
    df_meta.to_csv(OUTPUT_DIR / "metadata.csv", index=False)
    
    # 2. Extract Top Keys
    print("Extracting top keys...")
    df_keys = db.execute_query(queries.TOP_KEYS_QUERY)
    df_keys.to_csv(OUTPUT_DIR / "top_10_keys.csv", index=False)
    
    # 3. Extract Top Tags
    print("Extracting top tags...")
    df_tags = db.execute_query(queries.TOP_TAGS_QUERY)
    df_tags.to_csv(OUTPUT_DIR / "top_10_tags.csv", index=False)
    
    print("Analysis complete! Results saved to the 'output' directory.")

if __name__ == "__main__":
    main()
