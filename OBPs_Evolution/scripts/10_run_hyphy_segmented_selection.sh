#!/usr/bin/env bash
set -euo pipefail

MANIFEST="results/09_gard_partitions/segment_summary.tsv"

[[ -f "$MANIFEST" ]] || {
  echo "Missing $MANIFEST. Run python3 scripts/08_prepare_fixed_gard_partitions.py first." >&2
  exit 1
}

mkdir -p results/11_hyphy_segmented_selection

while IFS=$'\t' read -r segment_id alignment_path nt_start nt_end nt_length codon_start codon_end codon_length; do
  tree_path="results/10_gard_segment_trees/${segment_id}.treefile"
  [[ -f "$tree_path" ]] || {
    echo "Missing $tree_path. Run bash scripts/09_build_gard_segment_trees.sh first." >&2
    exit 1
  }

  outdir="results/11_hyphy_segmented_selection/${segment_id}"
  mkdir -p "$outdir"

  hyphy meme \
    --alignment "$alignment_path" \
    --tree "$tree_path" \
    --branches All \
    --pvalue 0.1 \
    --code Universal \
    --resample 100 \
    --output "${outdir}/meme.json" >"${outdir}/meme.log" </dev/null

  hyphy fel \
    --alignment "$alignment_path" \
    --tree "$tree_path" \
    --branches All \
    --pvalue 0.1 \
    --code Universal \
    --resample 100 \
    --output "${outdir}/fel.json" >"${outdir}/fel.log" </dev/null

  echo "Wrote ${outdir}/meme.json"
  echo "Wrote ${outdir}/fel.json"
done < <(tail -n +2 "$MANIFEST")
