#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/04_iqtree3

iqtree3 \
  -s results/03_pal2nal/codon.aln.fasta \
  --seqtype CODON \
  --prefix results/04_iqtree3/obp_codon \
  -m MFP \
  -T 22 \
  --seed 12345 \
  --redo

echo "Wrote results/04_iqtree3/obp_codon.treefile"
