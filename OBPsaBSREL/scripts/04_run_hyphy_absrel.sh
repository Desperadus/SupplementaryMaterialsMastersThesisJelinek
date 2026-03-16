#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/05_hyphy_absrel

hyphy absrel \
  --alignment results/03_pal2nal/codon.aln.fasta \
  --tree results/04_iqtree3/obp_codon.treefile \
  --branches All \
  --code Universal \
  --output results/05_hyphy_absrel/absrel.json >results/05_hyphy_absrel/absrel.log

echo "Wrote results/05_hyphy_absrel/absrel.json"
echo "Wrote results/05_hyphy_absrel/absrel.log"
