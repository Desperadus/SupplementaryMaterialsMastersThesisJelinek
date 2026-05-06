#!/usr/bin/env bash
set -euo pipefail

MANIFEST="results/09_gard_partitions/segment_summary.tsv"
ANALYSIS="/home/tomgolf/.conda/envs/evo/share/hyphy/TemplateBatchFiles/SelectionAnalyses/SingleOmega.bf"

[[ -f "$MANIFEST" ]] || {
  echo "Missing $MANIFEST. Run python3 scripts/08_prepare_fixed_gard_partitions.py first." >&2
  exit 1
}

[[ -f "$ANALYSIS" ]] || {
  echo "Missing HyPhy analysis batch file: $ANALYSIS" >&2
  exit 1
}

mkdir -p results/12_hyphy_fitmg94_segments

while IFS=$'\t' read -r segment_id alignment_path nt_start nt_end nt_length codon_start codon_end codon_length; do
  tree_path="results/10_gard_segment_trees/${segment_id}.treefile"
  [[ -f "$tree_path" ]] || {
    echo "Missing $tree_path. Run bash scripts/09_build_gard_segment_trees.sh first." >&2
    exit 1
  }

  default_json="${alignment_path}.SINGLE_OMEGA.json"
  result_json="results/12_hyphy_fitmg94_segments/${segment_id}.json"
  result_log="results/12_hyphy_fitmg94_segments/${segment_id}.log"

  rm -f "$default_json"

  hyphy "$ANALYSIS" \
    Universal \
    "$alignment_path" \
    "$tree_path" \
    All >"$result_log" </dev/null

  [[ -s "$default_json" ]] || {
    echo "HyPhy SingleOmega did not produce $default_json" >&2
    exit 1
  }

  mv -f "$default_json" "$result_json"

  echo "Wrote $result_json"
  echo "Wrote $result_log"
done < <(tail -n +2 "$MANIFEST")
