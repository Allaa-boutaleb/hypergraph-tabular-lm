import os
import subprocess
from pathlib import Path

def prepare_data():
    """
    Prepare SANTOS data following HyTrel's pipeline:
    1. Convert CSVs to JSONL chunks
    2. Clean and serialize to Arrow format
    """
    # Create directory structure
    base_dir = Path("data/santos")
    (base_dir / "chunks").mkdir(parents=True, exist_ok=True)
    (base_dir / "arrow").mkdir(parents=True, exist_ok=True)
    
    print("Step 1: Converting CSV files to JSONL chunks...")
    subprocess.run(["python", "csv_to_jsonl.py"], check=True)
    
    print("\nStep 2: Running parallel_clean.py to create Arrow files...")
    subprocess.run(["python", "parallel_clean.py"], check=True)
    
    print("\nData preparation complete!")

if __name__ == "__main__":
    prepare_data()