#!/bin/bash

benchmarks=(
    "pylon"
    # "pylon-p-both"
    "pylon-p-col"
    # "pylon-p-row"
    "santos"
    # "santos-p-both"
    "santos-p-col"
    # "santos-p-row"
    "tus"
    # "tus-p-both"
    "tus-p-col"
    # "tus-p-row"
    "tusLarge"
    # "tusLarge-p-both"
    "tusLarge-p-col"
    # "tusLarge-p-row"
)

for benchmark in "${benchmarks[@]}"; do
    echo "Processing $benchmark..."
    echo "Step 1: Converting CSV to JSONL..."
    python csv_to_jsonl.py "$benchmark"
    
    echo "Step 2: Running parallel clean..."
    python parallel_clean.py "$benchmark"
    
    echo "Completed $benchmark"
    echo "-------------------"
done

echo "All benchmarks processed!"