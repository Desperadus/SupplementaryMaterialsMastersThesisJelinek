## Dependencies
- `seaborn, meeko, RDKit, autodock-vina`
or ideally install the deps using mamba from the `environment.yaml` file.


## Preparation
Put OBP5.pdb into the `protein_target` folder and run the `prepare_protein.sh` script.

Prepare the ligands by running:
```bash
python3 prepare_ligands.py
```

## The Docking
Adjust the docking grid coordinates in vina_config.txt and then run:
```
vina --config vina_config.txt
```
