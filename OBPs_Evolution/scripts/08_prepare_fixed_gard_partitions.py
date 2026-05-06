#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

from Bio import SeqIO


ROOT = Path(__file__).resolve().parents[1]
CODON_ALIGNMENT = ROOT / "results/03_pal2nal/codon.aln.fasta"
OUTPUT_DIR = ROOT / "results/09_gard_partitions"
SEGMENTS_DIR = OUTPUT_DIR / "segments"
SUMMARY_TSV = OUTPUT_DIR / "segment_summary.tsv"
REFERENCE_TXT = OUTPUT_DIR / "gard_reference.txt"

# These fixed boundaries come from the project 34 GARD result.
BREAKPOINTS_NT = [165, 247, 374]
BEST_MODEL_AICC = 3724.280274970716
SINGLE_TREE_AICC = 3726.900523540161


def main() -> None:
    records = list(SeqIO.parse(CODON_ALIGNMENT, "fasta"))
    if not records:
        raise SystemExit(f"No sequences found in {CODON_ALIGNMENT}")

    lengths = {len(record.seq) for record in records}
    if len(lengths) != 1:
        raise SystemExit(f"Inconsistent alignment lengths: {sorted(lengths)}")

    alignment_length = lengths.pop()
    if alignment_length % 3 != 0:
        raise SystemExit(f"Alignment length is not divisible by 3: {alignment_length}")

    boundaries = [0, *BREAKPOINTS_NT, alignment_length]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)

    with SUMMARY_TSV.open("w") as handle:
        handle.write(
            "segment_id\talignment_path\tnt_start\tnt_end\tnt_length\tcodon_start\tcodon_end\tcodon_length\n"
        )

        for index, (start_0based, end_1based) in enumerate(zip(boundaries[:-1], boundaries[1:]), start=1):
            nt_start = start_0based + 1
            nt_end = end_1based
            nt_length = nt_end - nt_start + 1
            codon_start = ((nt_start - 1) // 3) + 1
            codon_end = nt_end // 3
            codon_length = codon_end - codon_start + 1

            if nt_length % 3 != 0:
                raise SystemExit(f"Segment {index} length is not divisible by 3: {nt_length}")

            segment_name = (
                f"segment_{index:02d}_nt{nt_start:03d}-{nt_end:03d}_codon{codon_start:03d}-{codon_end:03d}.fasta"
            )
            segment_path = SEGMENTS_DIR / segment_name

            segment_records = []
            for record in records:
                sliced = record[start_0based:nt_end]
                sliced.id = record.id
                sliced.name = record.name
                sliced.description = ""
                segment_records.append(sliced)
            SeqIO.write(segment_records, segment_path, "fasta")

            handle.write(
                "\t".join(
                    [
                        f"segment_{index:02d}",
                        str(segment_path.relative_to(ROOT)),
                        str(nt_start),
                        str(nt_end),
                        str(nt_length),
                        str(codon_start),
                        str(codon_end),
                        str(codon_length),
                    ]
                )
                + "\n"
            )

    REFERENCE_TXT.write_text(
        "\n".join(
            [
                "Fixed GARD reference values used in supplementary materials",
                f"best_model_aicc\t{BEST_MODEL_AICC}",
                f"single_tree_aicc\t{SINGLE_TREE_AICC}",
                "breakpoints_nt\t165,247,374",
                "segments_nt\t1-165,166-246,247-375,376-486",
            ]
        )
        + "\n"
    )

    print(f"Wrote {SUMMARY_TSV.relative_to(ROOT)}")
    print(f"Wrote {REFERENCE_TXT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
