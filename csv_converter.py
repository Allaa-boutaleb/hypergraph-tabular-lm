import os
import json
import pandas as pd
import pyarrow as pa
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class TableHeader:
    """Structure representing a table column header"""
    name: str
    type: str = "text"  # Default type
    is_primary_key: bool = False
    foreign_key: None = None
    
class CSVConverter:
    """Converts CSV files to HyTrel's expected format"""
    
    def __init__(self, max_rows: int = 30, max_cols: int = 20):
        self.max_rows = max_rows
        self.max_cols = max_cols

    def _infer_column_type(self, series: pd.Series) -> str:
        """Infer column type from data"""
        if pd.api.types.is_numeric_dtype(series):
            return "real"
        return "text"

    def _process_header(self, name: str, sample_value: Any) -> Dict:
        """Create header object in HyTrel format"""
        return {
            "name": str(name),
            "name_tokens": None,
            "type": self._infer_column_type(sample_value),
            "sample_value": {
                "value": str(sample_value),
                "tokens": str(sample_value).split(),
                "ner_tags": ["" for _ in str(sample_value).split()]
            },
            "sample_value_tokens": None,
            "is_primary_key": False,
            "foreign_key": None
        }

    def csv_to_jsonl(self, input_dir: str, output_dir: str):
        """Convert directory of CSV files to JSONL format"""
        os.makedirs(output_dir, exist_ok=True)
        
        jsonl_records = []
        
        for filename in os.listdir(input_dir):
            if not filename.endswith('.csv'):
                continue
                
            try:
                # Read CSV file
                df = pd.read_csv(os.path.join(input_dir, filename))
                
                # Truncate to max dimensions
                df = df.iloc[:self.max_rows, :self.max_cols]
                
                # Process headers
                headers = []
                for col in df.columns:
                    sample_value = df[col].iloc[0] if len(df) > 0 else ""
                    header = self._process_header(col, sample_value)
                    headers.append(header)
                
                # Convert data to list format
                data = df.values.tolist()
                
                # Create table record
                table_record = {
                    "id": filename,
                    "table": {
                        "caption": os.path.splitext(filename)[0],
                        "header": headers,
                        "data": data
                    },
                    "context_before": [],
                    "context_after": []
                }
                
                jsonl_records.append(table_record)
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
        
        # Write to JSONL file
        output_file = os.path.join(output_dir, "tables.jsonl")
        with open(output_file, 'w') as f:
            for record in jsonl_records:
                f.write(json.dumps(record) + '\n')

    def jsonl_to_arrow(self, jsonl_path: str, output_dir: str):
        """Convert JSONL to Arrow format using HyTrel's special tags"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Constants from HyTrel
        CAP_TAG = "<caption>"
        HEADER_TAG = "<header>"
        ROW_TAG = "<row>"
        
        texts = []
        
        # Read JSONL file
        with open(jsonl_path, 'r') as f:
            for line in f:
                record = json.loads(line)
                table = record['table']
                
                # Convert to HyTrel's text format
                text = f"{CAP_TAG} {table['caption']} "
                
                # Add headers
                headers = [h['name'] for h in table['header']]
                text += f"{HEADER_TAG} {' | '.join(headers)} "
                
                # Add rows
                for row in table['data']:
                    text += f"{ROW_TAG} {' | '.join(map(str, row))} "
                
                texts.append(text.strip())
        
        # Create Arrow table
        schema = pa.schema({'text': pa.large_string()})
        arr = pa.array(texts, type=pa.large_string())
        batch = pa.RecordBatch.from_arrays([arr], schema=schema)
        table = pa.Table.from_batches([batch], schema=schema)
        
        # Save Arrow file
        output_path = os.path.join(output_dir, "dataset.arrow")
        with pa.OSFile(output_path, 'wb') as sink:
            with pa.RecordBatchFileWriter(sink, schema) as writer:
                writer.write_batch(batch)