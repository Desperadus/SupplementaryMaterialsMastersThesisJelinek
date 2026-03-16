#!/usr/bin/env bash
set -euo pipefail
mkdir -p results/02_mafft

mafft \
  --auto \
  --thread 12 \
  --inputorder \
  results/01_sequences/protein.fasta \
  >results/02_mafft/protein.aln.fasta
