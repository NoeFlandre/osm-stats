import pandas as pd
from pathlib import Path
from src.database import OSMDatabase

class DataExporter:
    def __init__(self, db: OSMDatabase, output_dir: str | Path):
        self.db = db
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def export_query(self, query: str, filename: str) -> pd.DataFrame:
        """
        Executes a query and exports the resulting DataFrame to a CSV file.
        Returns the DataFrame that was exported.
        """
        df = self.db.execute_query(query)
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        return df
