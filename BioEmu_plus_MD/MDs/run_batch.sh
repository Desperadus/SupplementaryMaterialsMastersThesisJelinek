#!/bin/bash

PYTHON_SCRIPT="run_mds.py"
OUTPUT_BASE="md_results"

# Input Argument - desired dir of cluster centers
INPUT_DIR="$1"

if [ -z "$INPUT_DIR" ]; then
  echo "Usage: ./run_batch.sh <path_to_pdb_directory>"
  exit 1
fi

if [ ! -d "$INPUT_DIR" ]; then
  echo "Error: Directory '$INPUT_DIR' does not exist."
  exit 1
fi

if [ ! -f "$PYTHON_SCRIPT" ]; then
  echo "Error: $PYTHON_SCRIPT not found in the current directory."
  exit 1
fi

mkdir -p "$OUTPUT_BASE"

echo "Looking for files in: $INPUT_DIR"

for pdb_path in "$INPUT_DIR"/cluster_center_*.pdb; do

  if [ ! -e "$pdb_path" ]; then
    echo "No files matching 'cluster_center_*.pdb' found in $INPUT_DIR"
    break
  fi

  filename=$(basename "$pdb_path")
  cluster_dir=$(echo "$filename" | grep -o "cluster_center_[0-9]*")

  output_path="$OUTPUT_BASE/$cluster_dir"

  echo "========================================"
  echo "Processing: $filename"
  echo "Output Dir: $output_path"
  echo "========================================"

  python "$PYTHON_SCRIPT" \
    --input-pdb "$pdb_path" \
    --output-dir "$output_path" \
    --prod-ns 50.0 \
    --equil-ns 0.5 \
    --padding 1.0 || echo "WARNING: Simulation failed for $filename"

  echo ""
done

echo "Batch processing complete. Yuhoooo"
