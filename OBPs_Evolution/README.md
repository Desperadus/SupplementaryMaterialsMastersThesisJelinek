# How to run

Put the nucleotide sequences from GeneBank of the genes: KJ605390.1, KJ605394.1, KJ605393.1, KJ605392.1, KJ605391.1, NM_001310328.1 into `results/01_sequences/nucleotide_cds.fasta` file and the protein sequences into `results/01_sequences/protein.fasta`.

Should look like this then:

```
==> results/01_sequences/nucleotide_cds.fasta <==
>KJ605390.1 lcl|KJ605390.1_cds_AIA65156.1_1 [gene=OBP1] [protein=odorant binding protein 1] [protein_id=AIA65156.1] [location=29..550] [gbkey=CDS]
ATGGTGAAGTTCCTGCTAATTGCGCTTGCATTAGGTGTATCCTGTGCACATCATGAATCT
CTTGATATCAGTCCCTCAGAGATTGATGGGAACTGGCGCACATTTTACATAGCTGCGGAC
.
.
.

==> results/01_sequences/protein.fasta <==
>KJ605390.1 lcl|KJ605390.1_prot_AIA65156.1_1 [gene=OBP1] [protein=odorant binding protein 1] [protein_id=AIA65156.1] [location=29..550] [gbkey=CDS]
MVKFLLIALALGVSCAHHESLDISPSEIDGNWRTFYIAADKEEKVKMNGALRAYFEHMEC
NDDCGTLKIKFHVQMNGKCQTHTVVGEKQEDGRYTTDYSGRNYFEVVRKKDGALFFHNVN
.
.
.
```


Also put the `pal2nal.pl` from https://github.com/liaochenlanruo/PAL2NAL here into this folder where this README resides.

Then you can run:
`./scripts/01_align_proteins.sh`
`./scripts/02_build_codon_alignment.sh`
`./scripts/03_build_tree.sh`
`./scripts/04_run_hyphy_absrel.sh`
`./scripts/05_run_hyphy_meme.sh`
`./scripts/06_run_hyphy_fel.sh`
`./scripts/07_run_hyphy_gard.sh`
`python3 ./scripts/08_prepare_fixed_gard_partitions.py`
`./scripts/09_build_gard_segment_trees.sh`
`./scripts/10_run_hyphy_segmented_selection.sh`
`./scripts/11_run_hyphy_fitmg94_segments.sh`

- The results will appear in the `results` folder.

If you wish to run everything - run the `./run_all.sh` file.
