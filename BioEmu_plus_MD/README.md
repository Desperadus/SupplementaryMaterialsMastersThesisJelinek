## Dependencies
- `BioEmu[md], OpenMM, biopython, typer, scipy, mdcovert, xtb, mdtraj, numpy, deeptime, seaborn`

## Running

#### BioEmu + filtering section

First put a musM_OBP5_mature.fasta into this folder.

Then sample the BioEmu poses:
```bash
python3 run_bioemu.py
```

Then reconstruct sidechains using Hpacker:
```bash
python -m bioemu.sidechain_relax --pdb-path bioemu_output/topology.pdb --xtc-path bioemu_output/samples.xtc
```

Then inside the `bioemu_output` directory extract the structures to .pdb file from the .xtc (we used `mdconvert samples_sidechain_rec.xtc -o frames/output.pdb -t samples_sidechain_rec.pdb`) into the `frames` folder. And then turned the `output.pdb` into individual .pdb files using obabel (`obabel output.pdb -O frame.pdb -m`)

Then call inside the `bioemu_output` folder:
```bash
python3 optimize2.py --input-path frames --output-dir relaxed_structures
```
To form the disulfide bridges and relax the structures.

Then you can inside the `relaxed_structures` folder call the `get_energies.sh` script, which calculates the energy of the structure using GFN-FF. So you can filter out bad conformations.

#### Clustering

In the `clustering` folder run the `cluster_bioemu2.py` script with input folder being the filter (good) structures.

#### Running the MDs

Run the `run_batch.sh` script and provide as a argument path to a directory where you store the pdb files of cluster centers.

#### Analysis
Now its upto *you* to analyze the data - you can view our notebooks which we have used to analyze our outputs - and show the failed convergence :(. They are located in the `MDs/md_results` folder.

#### Project architecture
```
├── bioemu_output
│   ├── frames
│   ├── optimize2.py
│   └── relaxed_structures
│       └── get_energies.sh
├── clustering
│   └── cluster_bioemu2.py
├── MDs
│   ├── md_results
│   │   ├── analysis3.ipynb
│   │   └── projection.ipynb
│   ├── run_batch.sh
│   └── run_mds.py
├── README.md
└── run_bioemu.py

```
