#!/usr/bin/env python3

import sys
import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime

import typer
import openmm as mm
import openmm.app as app
import openmm.unit as unit
from openmmplumed import PlumedForce
from openmm.app import PDBFile, Modeller, ForceField, StateDataReporter, CheckpointReporter, DCDReporter
from openmm import LangevinMiddleIntegrator, MonteCarloBarostat, XmlSerializer
from pdbfixer import PDBFixer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("obp5-metad")

app_cli = typer.Typer(help="mOBP5 Metadynamics Workflow using OpenMM and PLUMED.")


def _dt_ps_from_fs(timestep_fs: float) -> float:
    # 1 ps = 1000 fs
    return timestep_fs / 1000.0


def _steps_from_time(length_ns: float, timestep_fs: float) -> int:
    dt_ps = _dt_ps_from_fs(timestep_fs)
    total_ps = length_ns * 1000.0
    return int(round(total_ps / dt_ps))


def _safe_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


class OBP5Metadynamics:
    def __init__(
        self,
        input_pdb: Path,
        output_dir: Path,
        temperature_k: float = 300.0,
        friction_ps: float = 1.0,
        timestep_fs: float = 2.0,
        padding_nm: float = 1.0,
        pressure_atm: float = 1.0,
        barostat_interval: int = 25,
        # Gate monitor definition (defaults: residue 24 CA to residue 73 CA)
        chain1: Optional[str] = None,
        res1: int = None,
        atom1: str = "CA",
        chain2: Optional[str] = None,
        res2: int = None,
        atom2: str = "CA",
        gyration_residues: str = "73,58,45,27,25,95",
        # Metadynamics params
        sigma: float = 0.05,
        height: float = 0.5,
        pace: int = 3000,
        biasfactor: float = 14.0,
        grid_min: float = 0.2,
        grid_max: float = 2.5,
        platform: Optional[str] = None,
        # PDBFixer options
        add_missing_residues: bool = True,
        add_missing_atoms: bool = True,
        remove_heterogens: bool = False,  # Keep existing waters by default (from Cavefiller)
        ph: float = 7.0,
        wall_at: float = 2.5,
        wall_kappa: float = 1500,
        lower_wall_at: float = 0.6,
        lower_wall_kappa: float = 1500,
    ):
        if not input_pdb.exists():
            raise FileNotFoundError(f"Input PDB not found at {input_pdb}")

        self.input_pdb = input_pdb
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.temperature_k = temperature_k
        self.friction_ps = friction_ps
        self.timestep_fs = timestep_fs

        self.padding_nm = padding_nm
        self.pressure_atm = pressure_atm
        self.barostat_interval = barostat_interval

        self.chain1, self.res1, self.atom1 = chain1, res1, atom1
        self.chain2, self.res2, self.atom2 = chain2, res2, atom2
        self.gyration_residues = gyration_residues

        self.sigma = sigma
        self.height = height
        self.pace = pace
        self.biasfactor = biasfactor
        self.grid_min = grid_min
        self.grid_max = grid_max

        self.platform_name = platform

        # PDBFixer options
        self.add_missing_residues = add_missing_residues
        self.add_missing_atoms = add_missing_atoms
        self.remove_heterogens = remove_heterogens
        self.ph = ph

        self.wall_at = wall_at
        self.wall_kappa = wall_kappa
        self.lower_wall_at = lower_wall_at
        self.lower_wall_kappa = lower_wall_kappa

        # Amber14SB + TIP3P
        self.forcefield = ForceField("amber14-all.xml", "amber14/tip3p.xml")

        self.system: Optional[mm.System] = None
        self.simulation: Optional[app.Simulation] = None
        self.integrator: Optional[LangevinMiddleIntegrator] = None
        self.modeller: Optional[Modeller] = None

        # Paths
        self.fixed_pdb_path = self.output_dir / "fixed.pdb"
        self.solvated_pdb_path = self.output_dir / "solvated.pdb"
        self.system_xml_path = self.output_dir / "system.xml"
        self.checkpoint_path = self.output_dir / "checkpoint.chk"
        self.traj_path = self.output_dir / "trajectory.dcd"
        self.state_log_path = self.output_dir / "state.log"
        self.plumed_used_path = self.output_dir / "plumed.used.dat"
        self.run_config_path = self.output_dir / "run_config.json"
        self.hills_path = (self.output_dir / "HILLS").resolve()

    def _stored_run_config(self) -> Optional[dict]:
        if not self.run_config_path.exists():
            return None
        return json.loads(self.run_config_path.read_text())

    def _write_run_config(self) -> None:
        config = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "input_pdb": str(self.input_pdb),
            "temperature_k": self.temperature_k,
            "friction_ps": self.friction_ps,
            "timestep_fs": self.timestep_fs,
            "padding_nm": self.padding_nm,
            "pressure_atm": self.pressure_atm,
            "barostat_interval": self.barostat_interval,
            "chain1": self.chain1,
            "res1": self.res1,
            "atom1": self.atom1,
            "chain2": self.chain2,
            "res2": self.res2,
            "atom2": self.atom2,
            "gyration_residues": self.gyration_residues,
            "sigma": self.sigma,
            "height": self.height,
            "pace": self.pace,
            "biasfactor": self.biasfactor,
            "grid_min": self.grid_min,
            "grid_max": self.grid_max,
            "wall_at": self.wall_at,
            "wall_kappa": self.wall_kappa,
            "lower_wall_at": self.lower_wall_at,
            "lower_wall_kappa": self.lower_wall_kappa,
            "platform": self.platform_name,
        }
        _safe_write_text(self.run_config_path, json.dumps(config, indent=2, sort_keys=True) + "\n")

    def _apply_restart_config(self) -> None:
        config = self._stored_run_config()
        if config is None:
            logger.warning(
                "Restart metadata not found at %s; using CLI integrator settings as-is.",
                self.run_config_path,
            )
            return

        critical_fields = ("temperature_k", "friction_ps", "timestep_fs")
        for field in critical_fields:
            current = getattr(self, field)
            stored = config.get(field)
            if stored is None:
                continue
            if current != stored:
                logger.warning(
                    "Restart requested with %s=%s, but the original run used %s. "
                    "Using the original value to preserve continuity.",
                    field,
                    current,
                    stored,
                )
                setattr(self, field, stored)

    def _current_step(self) -> int:
        if self.simulation is None:
            return 0
        return int(getattr(self.simulation, "currentStep", 0))

    def _current_time_ns(self) -> float:
        if self.simulation is None:
            return 0.0
        state = self.simulation.context.getState(getEnergy=False)
        time_ps = state.getTime().value_in_unit(unit.picoseconds)
        return float(time_ps) / 1000.0

    def _fix_pdb(self, input_path: Path, output_path: Path) -> Tuple[app.Topology, list]:
        logger.info(f"Running PDBFixer on {input_path}...")
        
        fixer = PDBFixer(filename=str(input_path))
        
        # Find missing residues
        fixer.findMissingResidues()
        if fixer.missingResidues:
            logger.info(f"Found {len(fixer.missingResidues)} missing residue ranges")
            if self.add_missing_residues:
                logger.info("Adding missing residues...")
            else:
                logger.info("Skipping missing residues (add_missing_residues=False)")
                fixer.missingResidues = {}
        
        # Find and replace non-standard residues
        fixer.findNonstandardResidues()
        if fixer.nonstandardResidues:
            logger.info(f"Found {len(fixer.nonstandardResidues)} non-standard residues")
            logger.info("Replacing non-standard residues with standard equivalents...")
            fixer.replaceNonstandardResidues()
        
        # Find missing atoms (heavy atoms)
        fixer.findMissingAtoms()
        if fixer.missingAtoms or fixer.missingTerminals:
            n_missing = sum(len(atoms) for atoms in fixer.missingAtoms.values())
            n_terminals = sum(len(atoms) for atoms in fixer.missingTerminals.values())
            logger.info(f"Found {n_missing} missing heavy atoms and {n_terminals} missing terminal atoms")
            if self.add_missing_atoms:
                logger.info("Adding missing atoms...")
                fixer.addMissingAtoms()
        
        # Remove heterogens (not done by default)
        if self.remove_heterogens:
            logger.info("Removing heterogens (keeping water)...")
            fixer.removeHeterogens(keepWater=True)
        else:
            logger.info("Keeping all heterogens including crystallographic waters")
        
        # Add hydrogens at specified pH
        logger.info(f"Adding hydrogens at pH {self.ph}...")
        fixer.addMissingHydrogens(self.ph)
        
        # Save the fixed structure
        with open(output_path, 'w') as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)
        logger.info(f"Saved fixed structure to {output_path}")
        
        return fixer.topology, fixer.positions

    def _find_residue(self, topology: app.Topology, chain_id: Optional[str], res_id: int):
        target = str(res_id)
        matches = []
        for chain in topology.chains():
            if chain_id is not None and chain.id != chain_id:
                continue
            for res in chain.residues():
                if res.id == target:
                    matches.append(res)

        if not matches:
            ids_preview = []
            for chain in topology.chains():
                if chain_id is not None and chain.id != chain_id:
                    continue
                for res in chain.residues():
                    ids_preview.append(res.id)
                    if len(ids_preview) >= 12:
                        break
                if len(ids_preview) >= 12:
                    break
            raise ValueError(
                f"Could not find residue id={target} (chain={chain_id}). "
                f"Preview of residue ids in selected chain(s): {ids_preview}"
            )

        if len(matches) > 1:
            raise ValueError(
                f"Residue id={target} matched multiple residues. "
                f"Specify --chain1/--chain2 to disambiguate."
            )
        return matches[0]

    def _plumed_atom_index(self, topology: app.Topology, chain_id: Optional[str], res_id: int, atom_name: str) -> int:
        """Return PLUMED atom index (1-based) that matches OpenMM Topology atom order."""
        res = self._find_residue(topology, chain_id, res_id)
        atom_matches = [a for a in res.atoms() if a.name == atom_name]
        if not atom_matches:
            names = [a.name for a in res.atoms()]
            raise ValueError(
                f"Could not find atom '{atom_name}' in residue id={res_id} (chain={chain_id}, resname={res.name}). "
                f"Atoms available in that residue: {names}"
            )
        if len(atom_matches) > 1:
            raise ValueError(
                f"Atom name '{atom_name}' appears multiple times in residue id={res_id} (chain={chain_id}). "
                f"Please choose a unique atom name."
            )
        # OpenMM Atom has .index (0-based in overall topology atom order)
        return atom_matches[0].index + 1

    def _parse_gyration_residue_ids(self) -> List[int]:
        tokens = [token for token in re.split(r"[\s,]+", self.gyration_residues.strip()) if token]
        if not tokens:
            raise ValueError("No residue ids were provided for the gyration CV.")

        residue_ids = []
        for token in tokens:
            try:
                residue_ids.append(int(token))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid gyration residue id '{token}' in '{self.gyration_residues}'. "
                    "Use a comma- or space-separated list like '73,58,45,27,25,95'."
                ) from exc

        return residue_ids

    def _sidechain_heavy_atom_indices(self, topology: app.Topology) -> List[int]:
        backbone_atoms = {"N", "CA", "C", "O", "OXT"}
        indices: List[int] = []

        for resid in self._parse_gyration_residue_ids():
            residue = self._find_residue(topology, None, resid)
            residue_indices = []
            for atom in residue.atoms():
                if atom.name in backbone_atoms:
                    continue
                if atom.element is not None and atom.element.symbol.upper() == "H":
                    continue
                residue_indices.append(atom.index + 1)

            if not residue_indices:
                raise ValueError(
                    f"Residue {residue.name}{residue.id} in chain {residue.chain.id} has no sidechain heavy atoms."
                )

            indices.extend(residue_indices)

        if not indices:
            raise ValueError("The gyration atom selection is empty.")

        return indices

    # ----------------------------
    # PLUMED
    # ----------------------------
    def _make_plumed_script(
        self,
        gyration_atoms: List[int],
        atom_i: int,
        atom_j: int,
        atom_n: int,
        atom_ca: int,
        atom_cb: int,
        atom_cg: int,
        restart: bool,
        colvar_path: Path,
    ) -> str:
        restart_line = "RESTART\n" if restart else ""
        hills = str(self.hills_path)

        colvar = str(colvar_path.resolve())
        gyration_atom_list = ",".join(str(idx) for idx in gyration_atoms)

        return f"""{restart_line}
# Sidechain heavy-atom gyration CV from residues listed in the input text file
gyr_group: GROUP ATOMS={gyration_atom_list}
sidechain_gyr: GYRATION TYPE=RADIUS ATOMS=gyr_group

# Tracked observables
gate_dist: DISTANCE ATOMS={atom_i},{atom_j}
chi1_res2: TORSION ATOMS={atom_n},{atom_ca},{atom_cb},{atom_cg}

# Well-Tempered Metadynamics
METAD ...
  LABEL=metad
  ARG=sidechain_gyr
  SIGMA={self.sigma}
  HEIGHT={self.height}
  PACE={self.pace}
  BIASFACTOR={self.biasfactor}
  TEMP={self.temperature_k}
  GRID_MIN={self.grid_min}
  GRID_MAX={self.grid_max}
  CALC_RCT
  FILE={hills}
... METAD

UPPER_WALLS ...
  LABEL=uwall
  ARG=sidechain_gyr
  AT={self.wall_at}
  KAPPA={self.wall_kappa}
  EXP=2
... UPPER_WALLS

LOWER_WALLS ...
  LABEL=lwall
  ARG=sidechain_gyr
  AT={self.lower_wall_at}
  KAPPA={self.lower_wall_kappa}
  EXP=2
... LOWER_WALLS

# Monitor CV, tracked observables, and instantaneous bias
PRINT ARG=sidechain_gyr,gate_dist,chi1_res2,metad.bias,metad.rbias,metad.rct,uwall.bias,lwall.bias STRIDE={self.pace} FILE={colvar}
"""

    def _select_platform(self) -> Optional[mm.Platform]:
        if not self.platform_name:
            return None
        try:
            return mm.Platform.getPlatformByName(self.platform_name)
        except Exception as e:
            raise ValueError(
                f"Requested platform '{self.platform_name}' not available. "
                f"Try 'CUDA', 'OpenCL', or 'CPU'. Original error: {e}"
            )

    def _build_integrator(self) -> LangevinMiddleIntegrator:
        return LangevinMiddleIntegrator(
            self.temperature_k * unit.kelvin,
            self.friction_ps / unit.picosecond,
            self.timestep_fs * unit.femtoseconds,
        )

    def _load_modeller(self, restart: bool) -> Modeller:
        if restart:
            if not self.solvated_pdb_path.exists():
                raise FileNotFoundError(
                    f"Restart requested but {self.solvated_pdb_path} not found. "
                    f"You need the saved solvated topology from the initial run."
                )
            pdb = PDBFile(str(self.solvated_pdb_path))
            return Modeller(pdb.topology, pdb.positions)

        # Fresh run: use PDBFixer to prepare the structure
        topology, positions = self._fix_pdb(self.input_pdb, self.fixed_pdb_path)
        return Modeller(topology, positions)

    def _solvate_if_needed(self, modeller: Modeller) -> Modeller:
        logger.info("Adding additional solvent/ions (preserving existing waters)...")
        
        n_waters_before = sum(1 for res in modeller.topology.residues() if res.name in ['HOH', 'WAT'])
        logger.info(f"Existing water molecules in structure: {n_waters_before}")
        
        modeller.addSolvent(
            self.forcefield,
            model="tip3p",
            padding=self.padding_nm * unit.nanometers,
            neutralize=True,
        )
        
        n_waters_after = sum(1 for res in modeller.topology.residues() if res.name in ['HOH', 'WAT'])
        n_waters_added = n_waters_after - n_waters_before
        logger.info(f"Added {n_waters_added} new water molecules (total: {n_waters_after})")
        
        with open(self.solvated_pdb_path, "w") as f:
            PDBFile.writeFile(modeller.topology, modeller.positions, f)
        logger.info(f"Wrote solvated topology to {self.solvated_pdb_path}")
        return modeller

    def _create_system_unbiased(self, topology: app.Topology) -> mm.System:
        # No barostat and no PLUMED here; unbiased equil first.
        system = self.forcefield.createSystem(
            topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=app.HBonds,
            rigidWater=True,
            hydrogenMass=1.5 * unit.amu,
        )
        return system

    def _add_barostat(self, system: mm.System) -> None:
        system.addForce(
            MonteCarloBarostat(
                self.pressure_atm * unit.atmospheres,
                self.temperature_k * unit.kelvin,
                self.barostat_interval,
            )
        )

    def _add_plumed(self, system: mm.System, plumed_script: str) -> None:
        system.addForce(PlumedForce(plumed_script))

    def setup(
        self,
        restart: bool = False,
        equil_nvt_ps: float = 100.0,
        equil_npt_ns: float = 1.0,
    ) -> None:
        if restart:
            self._apply_restart_config()

        platform = self._select_platform()
        self.integrator = self._build_integrator()

        if restart:
            if not self.system_xml_path.exists():
                raise FileNotFoundError(
                    f"Restart requested but {self.system_xml_path} not found."
                )

            self.modeller = self._load_modeller(restart=True)
            with open(self.system_xml_path, "r") as f:
                self.system = XmlSerializer.deserialize(f.read())

            if platform is None:
                self.simulation = app.Simulation(self.modeller.topology, self.system, self.integrator)
            else:
                self.simulation = app.Simulation(self.modeller.topology, self.system, self.integrator, platform)

            if not self.checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Restart requested but {self.checkpoint_path} not found. "
                    f"An exact continuation requires a saved checkpoint."
                )

            logger.info(f"Restoring from checkpoint {self.checkpoint_path}")
            self.simulation.loadCheckpoint(str(self.checkpoint_path))
            logger.info(
                "Restart continuation loaded at step %d (simulation time %.3f ns)",
                self._current_step(),
                self._current_time_ns(),
            )

            return  # done

        # Fresh run
        self.modeller = self._load_modeller(restart=False)
        self.modeller = self._solvate_if_needed(self.modeller)

        self.system = self._create_system_unbiased(self.modeller.topology)

        if platform is None:
            self.simulation = app.Simulation(self.modeller.topology, self.system, self.integrator)
        else:
            self.simulation = app.Simulation(self.modeller.topology, self.system, self.integrator, platform)

        self.simulation.context.setPositions(self.modeller.positions)

        logger.info("Minimizing energy...")
        self.simulation.minimizeEnergy()
        self.simulation.context.setVelocitiesToTemperature(self.temperature_k * unit.kelvin)

        nvt_steps = _steps_from_time(equil_nvt_ps / 1000.0, self.timestep_fs)  # ps -> ns
        if nvt_steps > 0:
            logger.info(f"Unbiased NVT warmup: {equil_nvt_ps} ps ({nvt_steps} steps)")
            self.simulation.step(nvt_steps)

        # --- Add barostat + NPT equilibration
        logger.info("Adding barostat and reinitializing context (preserve state)...")
        self._add_barostat(self.system)
        self.simulation.context.reinitialize(preserveState=True)

        npt_steps = _steps_from_time(equil_npt_ns, self.timestep_fs)
        if npt_steps > 0:
            logger.info(f"Unbiased NPT equilibration: {equil_npt_ns} ns ({npt_steps} steps)")
            self.simulation.step(npt_steps)

        gyration_atoms = self._sidechain_heavy_atom_indices(self.modeller.topology)
        atom_i = self._plumed_atom_index(self.modeller.topology, self.chain1, self.res1, self.atom1)
        atom_j = self._plumed_atom_index(self.modeller.topology, self.chain2, self.res2, self.atom2)
        atom_n = self._plumed_atom_index(self.modeller.topology, self.chain2, self.res2, "N")
        atom_ca = self._plumed_atom_index(self.modeller.topology, self.chain2, self.res2, "CA")
        atom_cb = self._plumed_atom_index(self.modeller.topology, self.chain2, self.res2, "CB")
        atom_cg = self._plumed_atom_index(self.modeller.topology, self.chain2, self.res2, "CG")
        logger.info(
            f"Gyration CV uses {len(gyration_atoms)} sidechain heavy atoms from residues "
            f"{self.gyration_residues}: "
            f"{gyration_atoms}"
        )
        logger.info(
            f"Gate monitor atoms (PLUMED 1-based indices): "
            f"({self.chain1 or '*'}:{self.res1}:{self.atom1}) -> {atom_i}, "
            f"({self.chain2 or '*'}:{self.res2}:{self.atom2}) -> {atom_j}"
        )
        logger.info(
            f"Chi1 CV atoms (PLUMED 1-based indices): "
            f"({self.chain2 or '*'}:{self.res2}:N-CA-CB-CG) -> {atom_n},{atom_ca},{atom_cb},{atom_cg}"
        )

        # Fresh run COLVAR path
        colvar_path = self.output_dir / "COLVAR"
        plumed_script = self._make_plumed_script(
            gyration_atoms=gyration_atoms,
            atom_i=atom_i,
            atom_j=atom_j,
            atom_n=atom_n,
            atom_ca=atom_ca,
            atom_cb=atom_cb,
            atom_cg=atom_cg,
            restart=False,
            colvar_path=colvar_path,
        )
        _safe_write_text(self.plumed_used_path, plumed_script)

        logger.info("Adding PLUMED force and reinitializing context (preserve state)...")
        self._add_plumed(self.system, plumed_script)
        self.simulation.context.reinitialize(preserveState=True)

        # Serialize full system (now includes barostat + PLUMED) for robust restart
        _safe_write_text(self.system_xml_path, XmlSerializer.serialize(self.system))
        logger.info(f"Wrote system XML to {self.system_xml_path}")
        self._write_run_config()
        logger.info(f"Wrote run configuration to {self.run_config_path}")

        self.simulation.saveCheckpoint(str(self.checkpoint_path))
        logger.info(f"Wrote initial checkpoint to {self.checkpoint_path}")

    # Production
    def run_metadynamics(self, length_ns: float, report_stride_steps: int = 50000) -> None:
        if self.simulation is None or self.system is None or self.modeller is None:
            raise RuntimeError("Call setup() before run_metadynamics().")

        steps = _steps_from_time(length_ns, self.timestep_fs)
        start_step = self._current_step()
        total_steps = start_step + steps
        start_time_ns = self._current_time_ns()

        resume = self.traj_path.exists()
        try:
            self.simulation.reporters.append(
                DCDReporter(str(self.traj_path), report_stride_steps, append=resume)
            )
        except TypeError:
            if resume:
                logger.warning("DCDReporter append not supported by your OpenMM; writing a new trajectory file.")
            self.simulation.reporters.append(DCDReporter(str(self.traj_path), report_stride_steps))

        self.simulation.reporters.append(
            StateDataReporter(
                sys.stdout,
                report_stride_steps,
                step=True,
                time=True,
                potentialEnergy=True,
                kineticEnergy=True,
                temperature=True,
                density=True,
                speed=True,
                progress=True,
                remainingTime=True,
                totalSteps=total_steps,
                separator="\t",
            )
        )
        self.simulation.reporters.append(
            StateDataReporter(
                str(self.state_log_path),
                report_stride_steps,
                step=True,
                time=True,
                potentialEnergy=True,
                kineticEnergy=True,
                temperature=True,
                density=True,
                speed=True,
                progress=True,
                remainingTime=True,
                totalSteps=total_steps,
                separator="\t",
                append=self.state_log_path.exists(),
            )
        )

        self.simulation.reporters.append(CheckpointReporter(str(self.checkpoint_path), report_stride_steps))

        logger.info(
            "Starting MetaD production from step %d / %.3f ns for an additional %.3f ns "
            "(%d steps, dt=%.3f fs), targeting step %d / %.3f ns",
            start_step,
            start_time_ns,
            length_ns,
            steps,
            self.timestep_fs,
            total_steps,
            start_time_ns + length_ns,
        )
        self.simulation.step(steps)
        self.simulation.saveCheckpoint(str(self.checkpoint_path))
        logger.info(
            "Production complete at step %d / %.3f ns; final checkpoint saved.",
            self._current_step(),
            self._current_time_ns(),
        )


@app_cli.command()
def main(
    input_pdb: Path = typer.Option(..., help="Initial (protein) PDB structure (can be unsolvated or have crystallographic waters)"),
    output_dir: Path = typer.Option(..., help="Directory for outputs"),
    ns: float = typer.Option(100.0, help="Production time (ns)"),
    restart: bool = typer.Option(False, "--restart", help="Resume using output_dir/solvated.pdb + system.xml + checkpoint.chk"),

    # Equilibration
    equil_nvt_ps: float = typer.Option(100.0, help="Unbiased NVT warmup (ps) before adding barostat (fresh run only)"),
    equil_npt_ns: float = typer.Option(1.0, help="Unbiased NPT equilibration (ns) before adding metadynamics (fresh run only)"),

    temperature_k: float = typer.Option(300.0, help="Temperature (K)"),
    friction_ps: float = typer.Option(1.0, help="Langevin friction (1/ps)"),
    timestep_fs: float = typer.Option(2.0, help="Timestep (fs). Start with 2.0; validate carefully before 4.0"),
    padding_nm: float = typer.Option(1.0, help="Solvent padding (nm) (fresh run only)"),
    pressure_atm: float = typer.Option(1.0, help="Pressure (atm) (fresh run only)"),
    barostat_interval: int = typer.Option(25, help="Barostat interval (steps) (fresh run only)"),
    platform: Optional[str] = typer.Option(None, help="OpenMM platform name (e.g., CUDA, OpenCL, CPU)"),

    add_missing_residues: bool = typer.Option(True, help="Add missing residues using PDBFixer"),
    add_missing_atoms: bool = typer.Option(True, help="Add missing heavy atoms using PDBFixer"),
    remove_heterogens: bool = typer.Option(False, help="Remove heterogens (keeps water regardless)"),
    ph: float = typer.Option(7.0, help="pH for adding hydrogens"),

    # Gate monitor specification -For replication of my results for OBP2 this needs to be changed to 35 and 83
    chain1: Optional[str] = typer.Option(None, help="Chain id for residue 1 (optional)"),
    res1: int = typer.Option(24, help="Residue id for gate atom 1 (PDB residue number)"),
    atom1: str = typer.Option("CA", help="Atom name for gate atom 1 (e.g., N)"),
    chain2: Optional[str] = typer.Option(None, help="Chain id for residue 2 (optional)"),
    res2: int = typer.Option(73, help="Residue id for gate atom 2 (PDB residue number)"),
    atom2: str = typer.Option("CA", help="Atom name for gate atom 2 (e.g., OH for Tyr)"),
    gyration_residues: str = typer.Option(
        "73,58,45,27,25,95",
        help="Comma- or space-separated residue ids whose sidechain heavy atoms define the gyration CV.",
    ),

    # Metadynamics params
    sigma: float = typer.Option(0.05, help="Gaussian sigma for the sidechain gyration CV (nm)"),
    height: float = typer.Option(1, help="Gaussian height (kJ/mol in PLUMED default units)"),
    pace: int = typer.Option(2000, help="Deposit stride (steps)"),
    biasfactor: float = typer.Option(12.0, help="Well-tempered bias factor"),
    grid_min: float = typer.Option(0.3, help="Metad grid min for the sidechain gyration CV (nm)"),
    grid_max: float = typer.Option(2, help="Metad grid max for the sidechain gyration CV (nm)"),

    wall_at: float = typer.Option(
        1.4, help="Upper wall position in nm for the sidechain gyration CV."
    ),
    wall_kappa: float = typer.Option(
        2000.0, help="Force constant for the upper wall (kJ/mol/nm^2)."
    ),
    lower_wall_at: float = typer.Option(
        0.1, help="Lower wall position in nm for the sidechain gyration CV."
    ),
    lower_wall_kappa: float = typer.Option(
        2000.0, help="Force constant for the lower wall (kJ/mol/nm^2)."
    ),
):
    runner = OBP5Metadynamics(
        input_pdb=input_pdb,
        output_dir=output_dir,
        temperature_k=temperature_k,
        friction_ps=friction_ps,
        timestep_fs=timestep_fs,
        padding_nm=padding_nm,
        pressure_atm=pressure_atm,
        barostat_interval=barostat_interval,
        chain1=chain1,
        res1=res1,
        atom1=atom1,
        chain2=chain2,
        res2=res2,
        atom2=atom2,
        gyration_residues=gyration_residues,
        sigma=sigma,
        height=height,
        pace=pace,
        biasfactor=biasfactor,
        grid_min=grid_min,
        grid_max=grid_max,
        platform=platform,
        add_missing_residues=add_missing_residues,
        add_missing_atoms=add_missing_atoms,
        remove_heterogens=remove_heterogens,
        ph=ph,
        wall_at=wall_at,
        wall_kappa=wall_kappa,
        lower_wall_at=lower_wall_at,
        lower_wall_kappa=lower_wall_kappa,
    )

    runner.setup(
        restart=restart,
        equil_nvt_ps=equil_nvt_ps,
        equil_npt_ns=equil_npt_ns,
    )
    runner.run_metadynamics(ns)


if __name__ == "__main__":
    app_cli()
