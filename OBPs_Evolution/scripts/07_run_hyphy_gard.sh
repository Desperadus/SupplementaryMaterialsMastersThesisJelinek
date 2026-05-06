#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/08_hyphy_gard

hyphy gard \
  "ENV=TOLERATE_NUMERICAL_ERRORS=1;" \
  --type nucleotide \
  --alignment results/03_pal2nal/codon.aln.fasta \
  --code Universal \
  --output results/08_hyphy_gard/gard.json \
  --output-lf results/08_hyphy_gard/best-gard >results/08_hyphy_gard/gard.log

echo "Wrote results/08_hyphy_gard/gard.json"
echo "Wrote results/08_hyphy_gard/best-gard"
echo "Wrote results/08_hyphy_gard/gard.log"
