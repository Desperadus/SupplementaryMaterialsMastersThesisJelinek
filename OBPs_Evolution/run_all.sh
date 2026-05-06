#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

bash scripts/01_align_proteins.sh
bash scripts/02_build_codon_alignment.sh
bash scripts/03_build_tree.sh
bash scripts/04_run_hyphy_absrel.sh
bash scripts/05_run_hyphy_meme.sh
bash scripts/06_run_hyphy_fel.sh
bash scripts/07_run_hyphy_gard.sh
python3 scripts/08_prepare_fixed_gard_partitions.py
bash scripts/09_build_gard_segment_trees.sh
bash scripts/10_run_hyphy_segmented_selection.sh
bash scripts/11_run_hyphy_fitmg94_segments.sh
