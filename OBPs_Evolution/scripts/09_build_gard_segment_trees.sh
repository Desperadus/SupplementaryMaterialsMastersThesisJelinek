#!/usr/bin/env bash
set -euo pipefail

MANIFEST="results/09_gard_partitions/segment_summary.tsv"

[[ -f "$MANIFEST" ]] || {
  echo "Missing $MANIFEST. Run python3 scripts/08_prepare_fixed_gard_partitions.py first." >&2
  exit 1
}

mkdir -p results/10_gard_segment_trees

while IFS=$'\t' read -r segment_id alignment_path nt_start nt_end nt_length codon_start codon_end codon_length; do
  prefix="results/10_gard_segment_trees/${segment_id}"
  iqtree3 \
    -s "$alignment_path" \
    --seqtype CODON \
    --prefix "$prefix" \
    -m MFP \
    -T 22 \
    --seed 12345 \
    --redo </dev/null

  echo "Wrote ${prefix}.treefile"
done < <(tail -n +2 "$MANIFEST")
