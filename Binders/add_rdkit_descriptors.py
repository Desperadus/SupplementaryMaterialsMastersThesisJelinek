#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from typing import Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors
import rdkit.Chem.rdFreeSASA as rdFreeSASA


def _nan_dict() -> dict[str, float]:
    return {
        "mol_wt": math.nan,
        "exact_mol_wt": math.nan,
        "logp": math.nan,
        "tpsa": math.nan,
        "hbd": math.nan,
        "hba": math.nan,
        "rotatable_bonds": math.nan,
        "fraction_csp3": math.nan,
        "mol_volume": math.nan,
        "sasa": math.nan,
    }


def calc_descriptors(smiles: Any) -> dict[str, float]:
    if not isinstance(smiles, str) or not smiles.strip():
        return _nan_dict()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return _nan_dict()

    out = {
        "mol_wt": Descriptors.MolWt(mol),
        "exact_mol_wt": Descriptors.ExactMolWt(mol),
        "logp": Crippen.MolLogP(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "hbd": float(Lipinski.NumHDonors(mol)),
        "hba": float(Lipinski.NumHAcceptors(mol)),
        "rotatable_bonds": float(Lipinski.NumRotatableBonds(mol)),
        "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
        "mol_volume": math.nan,
        "sasa": math.nan,
    }

    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    status = AllChem.EmbedMolecule(mol_h, params)
    if status != 0:
        return out

    try:
        AllChem.UFFOptimizeMolecule(mol_h, maxIters=200)
    except Exception:
        pass

    try:
        out["mol_volume"] = AllChem.ComputeMolVolume(mol_h)
    except Exception:
        pass

    try:
        radii = rdFreeSASA.classifyAtoms(mol_h)
        out["sasa"] = rdFreeSASA.CalcSASA(mol_h, radii)
    except Exception:
        pass

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Add RDKit descriptors to a CSV with a SMILES column.")
    parser.add_argument("-i", "--input", default="affinity_assay_moitrier_smiles.csv", help="Input CSV path")
    parser.add_argument(
        "-o",
        "--output",
        default="affinity_merged_with_rdkit_descriptors.csv",
        help="Output CSV path",
    )
    parser.add_argument("--smiles-col", default="smiles", help="SMILES column name")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.smiles_col not in df.columns:
        raise ValueError(f"Column '{args.smiles_col}' not found in {args.input}. Available: {list(df.columns)}")

    descriptors_df = df[args.smiles_col].apply(calc_descriptors).apply(pd.Series)
    result = pd.concat([df, descriptors_df], axis=1)
    result.to_csv(args.output, index=False)

    print(f"Wrote {len(result)} rows to {args.output}")


if __name__ == "__main__":
    main()
