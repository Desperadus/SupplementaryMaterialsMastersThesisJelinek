#!/bin/bash
echo "File, Energy" >>gfnff_en_summary.csv
for f in *_H.pdb; do
  echo "Calculating $f"
  energy=$(/home/tomgolf/.local/bin/xtb-dist/bin/xtb "$f" --gfnff --alpb water | grep "total energy" | awk '{print $4}')
  echo "$f, $energy" >>gfnff_en_summary.csv
done
