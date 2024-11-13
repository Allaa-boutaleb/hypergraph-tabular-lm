import os
import json
import pandas as pd
from tqdm import tqdm
from typing import Dict, List

class CSVtoJSONL:
    def __init__(self, chunk_size: int = 10):
        """
        Convert CSV files to JSONL format matching HyTrel's structure.
        Args:
            chunk_size: Number of tables per JSONL file
        """
        self.chunk_size = chunk_size
        
    def _process_column(self, name: str, sample_value: str) -> Dict:
        """Create column header metadata in HyTrel format"""
        return {
            "name": name,
            "name_tokens": name.split(),
            "type": "text",  # Default to text, could be enhanced with type inference
            "sample_value": {
                "value": str(sample_value),
                "tokens": str(sample_value).split(),
                "ner_tags": ["" for _ in str(sample_value).split()]
            },
            "sample_value_tokens": None,
            "is_primary_key": False,
            "foreign_key": None
        }

    def convert_csv_directory(self, input_dir: str, output_dir: str):
        """Convert directory of CSVs to chunked JSONL files"""
        os.makedirs(output_dir, exist_ok=True)
        tables = []
        chunk_id = 0
        
        print(f"Processing CSV files from {input_dir}...")
        for filename in tqdm(os.listdir(input_dir)):
            if not filename.endswith('.csv'):
                continue
                
            try:
                # Read CSV
                df = pd.read_csv(os.path.join(input_dir, filename))
                
                # Process headers
                headers = []
                for col in df.columns:
                    sample_val = df[col].iloc[0] if len(df) > 0 else ""
                    header = self._process_column(col, sample_val)
                    headers.append(header)
                
                # Convert data to list format, handling non-string data
                data = [[str(val) for val in row] for row in df.values.tolist()]
                
                # Create table record
                table = {
                    "id": filename,
                    "table": {
                        "caption": os.path.splitext(filename)[0],
                        "header": headers,
                        "data": data
                    },
                    "context_before": [],
                    "context_after": []
                }
                
                tables.append(table)
                
                # Write chunk if we've reached chunk_size
                if len(tables) >= self.chunk_size:
                    self._write_chunk(tables, output_dir, chunk_id)
                    tables = []
                    chunk_id += 1
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
        
        # Write remaining tables
        if tables:
            self._write_chunk(tables, output_dir, chunk_id)
    
    def _write_chunk(self, tables: List[Dict], output_dir: str, chunk_id: int):
        """Write a chunk of tables to JSONL file"""
        output_file = os.path.join(output_dir, f"chunk_{chunk_id:03d}.jsonl")
        with open(output_file, 'w') as f:
            for table in tables:
                f.write(json.dumps(table) + '\n')

if __name__ == "__main__":
    # Paths
    base_dir = "./data/"
    input_dir = os.path.join(base_dir, "santos/datalake")
    chunks_dir = os.path.join(base_dir, "pretrain/chunks")
    
    # Convert CSVs to JSONL chunks
    converter = CSVtoJSONL(chunk_size=10)
    converter.convert_csv_directory(input_dir, chunks_dir)
    print(f"\nCreated JSONL chunks in {chunks_dir}")
    print("Now run parallel_clean.py to create Arrow files")