## Metadynamics of OBP5 and OBP2

This provides the script to run a Molecular Metadynamics with two CVs the gate-dist and torsional angle. And also the script to then analyze the results and make the FES plots used in the thesis.

## Dependencies

Mamba environment used to run the simulation is provided as `environment.yaml`

## Running the simulation

Use:
`python3 run_md.py --input-pdb output/OBP5_filled_optimized.pdb --output-dir SIMULATION_RUN1 --ns 200`

To display possible flags use `--help`.

