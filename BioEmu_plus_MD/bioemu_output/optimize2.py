import os
import sys
import glob
import csv
from pathlib import Path
from enum import Enum
import typer
from tqdm import tqdm

from pdbfixer import PDBFixer
from openmm import app
import openmm as mm
from openmm import unit as u

DISULFIDE_PAIRS = [('34', '38'), ('53', '145')]
REPORT_CSV_NAME = "disulfide_distances.csv"

class MDProtocol(str, Enum):
    LOCAL_MINIMIZATION = "local_minimization"
    NVT_EQUIL = "nvt_equil"

def _find_sg_atom_index(topology: app.Topology, residue_id: str):
    """Return atom.index for SG in residue with res.id==residue_id, else None."""
    residue_map = {res.id: res for res in topology.residues()}
    res = residue_map.get(residue_id)
    if res is None:
        return None
    sg = next((a for a in res.atoms() if a.name == "SG"), None)
    return None if sg is None else sg.index

def compute_sg_sg_distances_angstrom(
    topology: app.Topology,
    positions,
    pairs: list[tuple[str, str]],
) -> dict[tuple[str, str], float | None]:
    """
    Compute SG-SG distances (Å) for each residue-id pair.
    Returns { (r1, r2): distance_A or None }.
    Safe: returns None if residues/SG atoms are missing.
    """
    try:
        pos_nm = positions.value_in_unit(u.nanometer)
    except Exception:
        pos_nm = positions

    out: dict[tuple[str, str], float | None] = {}
    for r1, r2 in pairs:
        i1 = _find_sg_atom_index(topology, r1)
        i2 = _find_sg_atom_index(topology, r2)
        if i1 is None or i2 is None:
            out[(r1, r2)] = None
            continue

        p1 = pos_nm[i1]
        p2 = pos_nm[i2]
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        dz = p1.z - p2.z
        dist_nm = (dx*dx + dy*dy + dz*dz) ** 0.5
        out[(r1, r2)] = dist_nm * 10.0  # nm -> A
    return out

def setup_disulfide_topology(
    pdb_path: str,
    pairs: list[tuple[str, str]],
    ph: float
) -> tuple[PDBFixer, dict]:
    """
    Loads PDB and forces disulfide bonds in the topology BEFORE adding hydrogens.
    Also measures SG-SG distances before adding bonds and after fixer processing.
    Returns (fixer, distance_report).
    """
    fixer = PDBFixer(filename=str(pdb_path))

    dist_initial = compute_sg_sg_distances_angstrom(fixer.topology, fixer.positions, pairs)

    residue_map = {res.id: res for res in fixer.topology.residues()}

    valid_pairs = []
    for r1_id, r2_id in pairs:
        if r1_id in residue_map and r2_id in residue_map:
            valid_pairs.append((r1_id, r2_id))
        else:
            print(f"  [Warning] Residues {r1_id} or {r2_id} missing in {pdb_path}. Skipping bond.")

    for r1_id, r2_id in valid_pairs:
        res1 = residue_map[r1_id]
        res2 = residue_map[r2_id]

        atom1 = next((a for a in res1.atoms() if a.name == 'SG'), None)
        atom2 = next((a for a in res2.atoms() if a.name == 'SG'), None)

        if atom1 and atom2:
            fixer.topology.addBond(atom1, atom2)

    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)

    dist_after_fixer = compute_sg_sg_distances_angstrom(fixer.topology, fixer.positions, pairs)

    report = {
        "initial_A": dist_initial,
        "after_fixer_A": dist_after_fixer,
    }
    return fixer, report

def run_constrained_md(
    fixer: PDBFixer,
    md_protocol: MDProtocol,
    pairs: list[tuple[str, str]],
    simtime_ns: float = 0.1,
    stiffness_k: float = 1000.0,
    constrain: bool = True
) -> tuple[app.Simulation, app.Modeller, dict]:
    """
    Sets up Explicit Solvent MD with Backbone Restraints.
    Returns (simulation, modeller, distance_report), where distance_report includes:
      - pre_optimization_A: SG-SG distances right before minimizeEnergy()
      - post_optimization_A: SG-SG distances after minimize (and after NVT if selected)
    """

    forcefield = app.ForceField("amber14/protein.ff14SB.xml", "amber14/tip3p.xml")

    modeller = app.Modeller(fixer.topology, fixer.positions)
    modeller.addSolvent(
        forcefield,
        padding=1.0 * u.nanometers,
        ionicStrength=0.1 * u.molar,
        positiveIon="Na+",
        negativeIon="Cl-",
    )

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * u.nanometers,
        constraints=app.HBonds
    )

    if constrain:
        force = mm.CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
        force.addGlobalParameter("k", stiffness_k)
        force.addPerParticleParameter("x0")
        force.addPerParticleParameter("y0")
        force.addPerParticleParameter("z0")

        print("Constraining the backbone")
        for atom in modeller.topology.atoms():
            if atom.residue.name not in ('HOH', 'NA', 'CL'):
                if atom.name in ("C", "CA", "N", "O"):
                    force.addParticle(atom.index, modeller.positions[atom.index])

        system.addForce(force)
    else:
        print("Did not constrain the backbone")

    integrator = mm.LangevinIntegrator(
        300.0 * u.kelvin,
        1.0 / u.picoseconds,
        0.002 * u.picoseconds
    )

    simulation = app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)

    dist_pre_opt = compute_sg_sg_distances_angstrom(modeller.topology, modeller.positions, pairs)

    simulation.minimizeEnergy(maxIterations=2000)

    if md_protocol == MDProtocol.NVT_EQUIL:
        simulation.context.setVelocitiesToTemperature(300.0 * u.kelvin)
        steps = int((simtime_ns * 1000) / 0.002)
        simulation.step(steps)

    state = simulation.context.getState(getPositions=True)
    dist_post_opt = compute_sg_sg_distances_angstrom(simulation.topology, state.getPositions(), pairs)

    report = {
        "pre_optimization_A": dist_pre_opt,
        "post_optimization_A": dist_post_opt,
    }
    return simulation, modeller, report

def save_stripped_pdb(simulation: app.Simulation, output_path: str):
    """
    Removes Solvent (HOH) and Ions (NA, CL) and saves the protein.
    """
    state = simulation.context.getState(getPositions=True)
    positions = state.getPositions()
    topology = simulation.topology

    modeller = app.Modeller(topology, positions)

    solvent_res_names = ['HOH', 'WAT', 'NA', 'CL', 'Na+', 'Cl-']
    to_delete = [r for r in modeller.topology.residues() if r.name in solvent_res_names]

    if len(to_delete) > 0:
        modeller.delete(to_delete)

    with open(output_path, 'w') as f:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, f, keepIds=True)

def _fmt_dist(x):
    return "NA" if x is None else f"{x:.3f}"

def main(
    input_path: str = typer.Option(..., help="Path to a single .pdb file OR a directory of .pdbs"),
    output_dir: str = typer.Option("relaxed_structures", help="Folder to save optimized PDBs"),
    md_protocol: MDProtocol = typer.Option(MDProtocol.LOCAL_MINIMIZATION, help="Protocol to run"),
    sim_time: float = typer.Option(0.1, help="Simulation time in ns (if NVT selected)"),
    ph: float = typer.Option(7, help="Protonation state for PDBFixer"),
    constrain: bool = typer.Option(
        False,
        "--constrain/--no-constrain",
        help="Whether to constrain the backbone",
    )
) -> None:

    input_p = Path(input_path)
    output_p = Path(output_dir)
    output_p.mkdir(parents=True, exist_ok=True)

    csv_path = output_p / REPORT_CSV_NAME
    new_csv = not csv_path.exists()
    csv_fh = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_fh)
    if new_csv:
        csv_writer.writerow([
            "pdb_file",
            "pair",
            "initial_distance_A",
            "after_fixer_distance_A",
            "pre_optimization_distance_A",
            "post_optimization_distance_A",
        ])

    # Gather files
    if input_p.is_dir():
        files_to_process = sorted(list(input_p.glob("*.pdb")))
    elif input_p.is_file():
        files_to_process = [input_p]
    else:
        print(f"Error: {input_path} is not a valid file or directory.")
        sys.exit(1)

    print(f"Found {len(files_to_process)} files to process.")
    print(f"Protocol: {md_protocol.value} | Output Dir: {output_dir}")
    print(f"Disulfide report CSV: {csv_path}")
    print("-" * 50)

    for pdb_file in tqdm(files_to_process, desc="Processing PDBs"):
        try:
            if constrain:
                out_name = pdb_file.stem + "_relaxed_and_bridge_formed_and_backbone_constrained.pdb"
            else:
                out_name = pdb_file.stem + "_relaxed_and_bridge_formed_and_backbone_unconstrained.pdb"
            out_path = output_p / out_name

            fixer, fixer_report = setup_disulfide_topology(str(pdb_file), DISULFIDE_PAIRS, ph)

            sim, _, md_report = run_constrained_md(
                fixer,
                md_protocol,
                DISULFIDE_PAIRS,
                simtime_ns=sim_time,
                constrain=constrain
            )

            save_stripped_pdb(sim, str(out_path))

            for (r1, r2) in DISULFIDE_PAIRS:
                initial = fixer_report["initial_A"].get((r1, r2))
                after_fixer = fixer_report["after_fixer_A"].get((r1, r2))
                pre_opt = md_report["pre_optimization_A"].get((r1, r2))
                post_opt = md_report["post_optimization_A"].get((r1, r2))

                print(
                    f"{pdb_file.name} [{r1}-{r2}] "
                    f"initial={_fmt_dist(initial)} Å | "
                    f"after_fixer={_fmt_dist(after_fixer)} Å | "
                    f"pre_opt={_fmt_dist(pre_opt)} Å | "
                    f"post_opt={_fmt_dist(post_opt)} Å"
                )

                csv_writer.writerow([
                    pdb_file.name,
                    f"{r1}-{r2}",
                    "" if initial is None else f"{initial:.6f}",
                    "" if after_fixer is None else f"{after_fixer:.6f}",
                    "" if pre_opt is None else f"{pre_opt:.6f}",
                    "" if post_opt is None else f"{post_opt:.6f}",
                ])
            csv_fh.flush()

        except Exception as e:
            print(f"\n[Error] Failed on {pdb_file.name}: {e}")
            continue

    csv_fh.close()
    print("-" * 50)
    print("Batch processing complete.")

if __name__ == "__main__":
    typer.run(main)

