import sys
import logging
from pathlib import Path
from typing import Optional, Tuple

import typer
import pandas as pd
import matplotlib.pyplot as plt
import openmm as mm
import openmm.app as app
import openmm.unit as unit
from openmm.app import PDBFile, Modeller, ForceField, StateDataReporter, CheckpointReporter, DCDReporter
from openmm import LangevinMiddleIntegrator, MonteCarloBarostat, XmlSerializer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app_cli = typer.Typer(help="Run simple BioEmu MD workflow (Min -> Equil -> Prod).")


class BioEmuSimpleMD:
    def __init__(self, pdb_path: Path, output_dir: Path):
        if not pdb_path.exists():
            raise FileNotFoundError(f"Input PDB not found at {pdb_path}")

        self.pdb_path = pdb_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)

        self.forcefield = ForceField("amber14-all.xml", "amber14/tip3p.xml")
        
        self.modeller: Optional[Modeller] = None
        self.system: Optional[mm.System] = None
        self.simulation: Optional[app.Simulation] = None
        self.integrator: Optional[LangevinMiddleIntegrator] = None

    def prepare_system(self, padding: float = 1.0, ionic_strength: float = 0.15):
        logger.info(f"Reading PDB: {self.pdb_path}")
        pdb = PDBFile(str(self.pdb_path))
        self.modeller = Modeller(pdb.topology, pdb.positions)

        logger.info(f"Solvating with {padding} nm padding...")
        
        self.modeller.addSolvent(
            self.forcefield,
            model="tip3p",
            padding=padding * unit.nanometers,
            neutralize=True,
            ionicStrength=ionic_strength * unit.molar,
            positiveIon='Na+',
            negativeIon='Cl-'
        )

        logger.info("Creating OpenMM System...")
        self.system = self.forcefield.createSystem(
            self.modeller.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=app.HBonds,
            rigidWater=True,
            ewaldErrorTolerance=0.0005,
            hydrogenMass=1.5 * unit.amu 
        )
        
        self.system.addForce(MonteCarloBarostat(1.0 * unit.atmospheres, 300.0 * unit.kelvin, 25))

        self.integrator = LangevinMiddleIntegrator(
            300.0 * unit.kelvin,
            1.0 / unit.picosecond,
            0.002 * unit.picoseconds
        )
        self.integrator.setConstraintTolerance(0.000001)

        self.simulation = app.Simulation(
            self.modeller.topology, self.system, self.integrator
        )
        self.simulation.context.setPositions(self.modeller.positions)
        with open(self.output_dir / "system.xml", "w") as f:
            f.write(XmlSerializer.serialize(self.system))
        with open(self.output_dir / "integrator.xml", "w") as f:
            f.write(XmlSerializer.serialize(self.integrator))

    def run_minimization(self):
        logger.info("--- Minimization ---")
        logger.info("Minimizing energy...")
        self.simulation.minimizeEnergy()
        
        min_pdb_path = self.output_dir / "minimized.pdb"
        with open(min_pdb_path, 'w') as f:
            PDBFile.writeFile(
                self.simulation.topology, 
                self.simulation.context.getState(getPositions=True).getPositions(), 
                f
            )
        logger.info(f"Minimization complete. Saved to {min_pdb_path}")

    def run_equilibration(self, length_ns: float = 0.5):
        logger.info(f"--- Equilibration ({length_ns} ns) ---")
        
        self.simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)
        
        steps = int((length_ns * 1000) / 0.002) # ns to ps to steps (dt=0.002ps)
        report_interval = steps // 100 if steps > 100 else 10 # 100 data points

        self.simulation.currentStep = 0

        self.simulation.reporters.append(
            StateDataReporter(
                str(self.output_dir / "equilibration_log.csv"),
                report_interval,
                step=True, time=True, potentialEnergy=True, 
                temperature=True, volume=True, density=True, 
                speed=True, separator=","
            )
        )
        
        logger.info(f"Stepping {steps} steps...")
        self.simulation.step(steps)
        
        eq_pdb_path = self.output_dir / "equilibrated.pdb"
        with open(eq_pdb_path, 'w') as f:
            PDBFile.writeFile(
                self.simulation.topology, 
                self.simulation.context.getState(getPositions=True).getPositions(), 
                f
            )
        logger.info(f"Equilibration complete. Saved to {eq_pdb_path}")
        
        self.simulation.reporters.clear()

    def run_production(self, length_ns: float = 2.0, dcd_frames: int = 1000):
        logger.info(f"--- Production ({length_ns} ns) ---")
        
        steps = int((length_ns * 1000) / 0.002)
        dcd_interval = int(steps / dcd_frames)
        log_interval = dcd_interval
        
        self.simulation.reporters.append(
            DCDReporter(str(self.output_dir / "trajectory.dcd"), dcd_interval)
        )
        
        self.simulation.reporters.append(
            StateDataReporter(
                str(self.output_dir / "production_log.csv"),
                dcd_interval,
                step=True, time=True, potentialEnergy=True, totalEnergy=True,
                temperature=True, volume=True, density=True, 
                speed=True, progress=True, remainingTime=True, separator=",",
                totalSteps=steps
            )
        )
        
        self.simulation.reporters.append(
            CheckpointReporter(
                str(self.output_dir / "checkpoint.chk"), 
                log_interval * 10
            )
        )

        logger.info(f"Running production for {steps} steps...")
        self.simulation.step(steps)
        
        self.simulation.saveState(str(self.output_dir / "final_state.xml"))
        logger.info("Production complete. Final state saved.")

    def generate_plots(self):
        logger.info("Generating QA plots...")
        
        for stage in ["equilibration_log", "production_log"]:
            csv_path = self.output_dir / f"{stage}.csv"
            if not csv_path.exists():
                continue
            
            try:
                df = pd.read_csv(csv_path)
                df.columns = df.columns.str.strip()                
                if 'Temperature (K)' in df.columns:
                    plt.figure()
                    plt.plot(df['Time (ps)'], df['Temperature (K)'], label='Temp')
                    plt.xlabel('Time (ps)')
                    plt.ylabel('Temperature (K)')
                    plt.title(f'{stage} - Temperature')
                    plt.savefig(self.plots_dir / f"{stage}_temperature.png")
                    plt.close()

                if 'Potential Energy (kJ/mole)' in df.columns:
                    plt.figure()
                    plt.plot(df['Time (ps)'], df['Potential Energy (kJ/mole)'], color='orange')
                    plt.xlabel('Time (ps)')
                    plt.ylabel('PE (kJ/mol)')
                    plt.title(f'{stage} - Potential Energy')
                    plt.savefig(self.plots_dir / f"{stage}_energy.png")
                    plt.close()

                if 'Density (g/mL)' in df.columns:
                    plt.figure()
                    plt.plot(df['Time (ps)'], df['Density (g/mL)'], color='green')
                    plt.xlabel('Time (ps)')
                    plt.ylabel('Density (g/mL)')
                    plt.title(f'{stage} - Density')
                    plt.savefig(self.plots_dir / f"{stage}_density.png")
                    plt.close()
            except Exception as e:
                logger.warning(f"Failed to plot {stage}: {e}")


@app_cli.command()
def main(
    input_pdb: Path = typer.Option(..., exists=True, dir_okay=False, help="Input PDB file"),
    output_dir: Path = typer.Option(..., file_okay=False, help="Output directory"),
    prod_ns: float = typer.Option(50, help="Production simulation length in ns"),
    equil_ns: float = typer.Option(0.5, help="Equilibration length in ns"),
    padding: float = typer.Option(1.0, help="Water box padding in nm"),
    frames: int = typer.Option(1000, help="Number of trajectory frames to save"),
):
    try:
        runner = BioEmuSimpleMD(input_pdb, output_dir)
        runner.prepare_system(padding=padding)
        runner.run_minimization()
        runner.run_equilibration(length_ns=equil_ns)
        runner.run_production(length_ns=prod_ns, dcd_frames=frames)
        runner.generate_plots()
        logger.info(f"Workflow completed successfully. Data in {output_dir}")

    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    app_cli()
