# OBP5 BioEmu Sampling Scripts

## Requirements

Put the samples.xtc and topology.pdb files from BioEmu here into this folder.

Python3 need these packages for plotting: `numpy, matplotlib, seaborn`

Further VMD program is needed.

## How to run

1. Run the Tcl scripts with VMD:

```bash
vmd -dispdev text -e <script_name>.tcl
```

2. Run the Python plotting scripts:

```bash
python3 <script_name>.py
```

The Tcl scripts generate the distance `.dat` files, and the Python scripts use those files to make the plots and CSV output.

