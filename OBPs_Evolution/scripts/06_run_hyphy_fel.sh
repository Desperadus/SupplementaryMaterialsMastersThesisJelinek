#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/07_hyphy_fel

hyphy fel \
  --alignment results/03_pal2nal/codon.aln.fasta \
  --tree results/04_iqtree3/obp_codon.treefile \
  --branches All \
  --pvalue 0.1 \
  --code Universal \
  --resample 100 \
  --output results/07_hyphy_fel/fel.json >results/07_hyphy_fel/fel.log

echo "Wrote results/07_hyphy_fel/fel.json"
echo "Wrote results/07_hyphy_fel/fel.log"
