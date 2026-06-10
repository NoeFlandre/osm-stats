import os
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
