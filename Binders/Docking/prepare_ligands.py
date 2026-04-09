import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation, PDBQTWriterLegacy

def prepare_ligands_meeko(input_csv, output_dir="ligands_pdbqt"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: Could not find {input_csv}")
        return

    print(f"Found {len(df)} compounds. Starting conversion with Meeko v0.5+...")
    print("-" * 60)

    for index, row in df.iterrows():
        name = str(row['Compound']).strip()
        smiles = str(row['smiles']).strip()
        
        safe_name = "".join([c if c.isalnum() or c in ('-','_') else "_" for c in name])
        output_file = os.path.join(output_dir, f"{safe_name}.pdbqt")

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Error: RDKit could not parse SMILES for {name}")
                continue

            mol_h = Chem.AddHs(mol)
            
            params = AllChem.ETKDGv3()
            params.useRandomCoords = True
            res = AllChem.EmbedMolecule(mol_h, params)
            
            if res == -1:
                print(f"Warning: Standard embedding failed for {name}, trying random coordinates...")
                AllChem.EmbedMolecule(mol_h, useRandomCoords=True)

            try:
                AllChem.UFFOptimizeMolecule(mol_h)
            except Exception:
                print("ERROR failed to UFFOptimizeMolecule")

            preparator = MoleculePreparation()
            mol_setups = preparator.prepare(mol_h)
            
            # print(PDBQTWriterLegacy.write_string(mol_setups[0]))
            pdbqt_string = PDBQTWriterLegacy.write_string(mol_setups[0])[0]

            with open(output_file, 'w') as f:
                f.write(pdbqt_string)

            print(f"Converted: {name}")

        except Exception as e:
            print(f"Failed to process {name}: {e}")

    print("-" * 60)
    print(f"Processing complete. Files saved in '{output_dir}/'")

if __name__ == "__main__":
    prepare_ligands_meeko("../affinity_assay_moitrier_smiles.csv")
