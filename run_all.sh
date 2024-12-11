#!/bin/bash

# echo "Starting data preparation..."
# ./prepare_data.sh

# echo "Starting pretraining process..."
# ./pretrain_contrast.sh

echo "Starting vector extraction..."
./extract_vectors.sh

echo "Starting file reorganizing..."
./reorganize_files.sh

echo "Starting benchmark evaluation..."
./evaluate.sh

echo "All processes completed!"