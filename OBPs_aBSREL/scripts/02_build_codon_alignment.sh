#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/03_pal2nal

perl pal2nal.pl \
  results/02_mafft/protein.aln.fasta \
  results/01_sequences/nucleotide_cds.fasta \
  -output fasta \
  -nogap \
  -nomismatch \
  -codontable 1 \
  >results/03_pal2nal/codon.aln.fasta

python3 - <<'PY'
from pathlib import Path
from Bio import SeqIO

path = Path("results/03_pal2nal/codon.aln.fasta")
records = list(SeqIO.parse(path, "fasta"))
if not records:
    raise SystemExit("PAL2NAL produced an empty alignment")
lengths = {len(record.seq) for record in records}
if len(lengths) != 1:
    raise SystemExit(f"Inconsistent codon alignment lengths: {sorted(lengths)}")
length = next(iter(lengths))
if length % 3 != 0:
    raise SystemExit(f"Codon alignment length is not divisible by 3: {length}")
print(f"Wrote {path}")
PY
