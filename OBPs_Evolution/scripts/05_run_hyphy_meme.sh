#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/06_hyphy_meme

hyphy meme \
  --alignment results/03_pal2nal/codon.aln.fasta \
  --tree results/04_iqtree3/obp_codon.treefile \
  --branches All \
  --pvalue 0.1 \
  --code Universal \
  --resample 100 \
  --output results/06_hyphy_meme/meme.json >results/06_hyphy_meme/meme.log

echo "Wrote results/06_hyphy_meme/meme.json"
echo "Wrote results/06_hyphy_meme/meme.log"
