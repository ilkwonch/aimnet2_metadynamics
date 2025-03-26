import argparse
import os
import time
import csv  # Added for CSV logging
from openmm import LangevinIntegrator, LangevinMiddleIntegrator, unit
from openmm.app import PDBFile, ForceField, Simulation, PDBReporter, StateDataReporter, DCDReporter, NoCutoff, HBonds
from openmm.app.metadynamics import Metadynamics, BiasVariable
from openmmml.mlpotential import MLPotential
import cv  
import numpy as np
from openmm.openmm import CustomTorsionForce
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from openmmml.mlpotential import MLPotential, MLPotentialImpl, MLPotentialImplFactory
from openmmml.models.aimnet2potential import AIMNet2PotentialImplFactory



def run_metadynamics_dipeptide_simulations(input_pdb, output_dir, steps, temperature, timestep, frequency, 
                                         system_type, sigma, forcefield_type, charge=None, model_path=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    energy_dir = os.path.join(output_dir, 'energy')
    if not os.path.exists(energy_dir):
        os.makedirs(energy_dir)

    pdb = PDBFile(input_pdb)
    
    # Choose force field based on user input
    if forcefield_type == "amber":
        forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    else:
        if forcefield_type == "aimnet2" and model_path:
            MLPotential.registerImplFactory('aimnet2', AIMNet2PotentialImplFactory())
            forcefield = MLPotential('aimnet2', 
                                  modelPath=model_path,
                                  charge=charge if charge is not None else 0.0)
        else:
            forcefield = MLPotential(forcefield_type)

    # Create system based on system_type
    if system_type == "gas":
        system = forcefield.createSystem(pdb.topology, nonbondedMethod=NoCutoff, constraints=HBonds)
    elif system_type == "solvent":
        system = forcefield.createSystem(pdb.topology, nonbondedCutoff=15*unit.angstrom, constraints=HBonds)
    else:
        raise ValueError(f"Unknown system type '{system_type}'.")

    integrator = LangevinMiddleIntegrator(temperature * unit.kelvin, 1.0 / unit.picosecond, timestep * unit.femtosecond)
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)

    print(f'Minimizing energy... MaxIterations: 100')
    simulation.minimizeEnergy(maxIterations=100)
    
    phi, psi = cv.get_phi_psi_indices(input_pdb)
    cv1 = CustomTorsionForce('theta')
    cv1.addTorsion(*phi)

    cv2 = CustomTorsionForce('theta')
    cv2.addTorsion(*psi)

    phi_cv = BiasVariable(cv1, -np.pi * unit.radian, np.pi * unit.radian, 
                          sigma * unit.radian, periodic=True)
    psi_cv = BiasVariable(cv2, -np.pi * unit.radian, np.pi * unit.radian, 
                          sigma * unit.radian, periodic=True)

    meta = Metadynamics(
        system=system,
        variables=[phi_cv, psi_cv],
        temperature=temperature * unit.kelvin,
        biasFactor=10.0,
        height=1.0 * unit.kilojoule_per_mole,
        frequency=500,
        saveFrequency=frequency,
        biasDir=energy_dir
    )
    
    integrator = LangevinMiddleIntegrator(temperature * unit.kelvin, 1.0 / unit.picosecond, timestep * unit.femtosecond)
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)

    # Add reporters
    simulation.reporters.append(PDBReporter(os.path.join(energy_dir, "snapshot.pdb"), frequency))
    simulation.reporters.append(StateDataReporter(
        os.path.join(output_dir, "state.csv"), frequency,
        step=True, potentialEnergy=True, temperature=True
    ))
    simulation.reporters.append(StateDataReporter(sys.stdout, 10, step=True, potentialEnergy=True, temperature=True))
    simulation.reporters.append(DCDReporter(os.path.join(energy_dir, "trajectory.dcd"), frequency))

    # Prepare CSV file for timing logs
    timing_file = os.path.join(output_dir, "step_timing.csv")
    with open(timing_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Step Range", "Steps", "Time (seconds)", "Time per Step (seconds)"])

    # Run metadynamics simulation with per-frequency timing
    print(f"Running {steps} steps of metadynamics simulation with forcefield: {forcefield_type}...")
    remaining_steps = steps
    current_step = 0

    while remaining_steps > 0:
        steps_to_run = min(frequency, remaining_steps)  # Run up to 'frequency' steps or remaining steps
        start_time = time.time()
        meta.step(simulation, steps_to_run)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        time_per_step = elapsed_time / steps_to_run
        
        # Log to CSV
        with open(timing_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([f"{current_step}-{current_step + steps_to_run}", steps_to_run, f"{elapsed_time:.4f}", f"{time_per_step:.6f}"])

        current_step += steps_to_run
        remaining_steps -= steps_to_run

    free_energy = meta.getFreeEnergy()
    np.savetxt(os.path.join(output_dir, "free_energy.txt"), free_energy)

    free_energy_array = np.array(free_energy)
    plt.imshow(free_energy_array, origin='lower', cmap='viridis', aspect='auto')
    plt.colorbar(label="Free Energy (kJ/mol)")
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$\psi$')
    plt.savefig(os.path.join(output_dir, "free_energy_landscape.png"))

    print(f"Simulation complete. Step timing saved to {timing_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run metadynamics simulation on a dipeptide")
    parser.add_argument('-i', '--input', type=str, required=True, help='Input PDB file')
    parser.add_argument('-o', '--output', type=str, default="alanine_dipeptide_meta", help='Output directory')
    parser.add_argument('-s', '--steps', type=int, default=50000, help='Number of simulation steps')
    parser.add_argument('-t', '--temperature', type=float, default=300, help='Simulation temperature (K)')
    parser.add_argument('-dt', '--timestep', type=float, default=2.0, help='Simulation timestep (fs)')
    parser.add_argument('-f', '--frequency', type=int, default=500, help='Reporting frequency (steps)')
    parser.add_argument('-sys', '--system', type=str, choices=['gas', 'solvent'], default='gas', help="Simulation system type")
    parser.add_argument('-sig', '--sigma', type=float, default=0.8, help="Bias coefficient")
    parser.add_argument('-ff', '--forcefield', type=str, default='amber', 
                        help="Force field type: 'amber' (default) or an MLPotential name (e.g., 'mace-off23-small/medium/large', 'nequip', 'ani2x','aimnet2')")
    parser.add_argument('--charge', type=float, default=None, 
                        help="Charge for AIMNET2 forcefield (default: 0.0 if not specified)")
    parser.add_argument('--model_path', type=str, default=None,
                        help="Path to AIMNET2 model file (required if using aimnet2 forcefield)")

    args = parser.parse_args()

    run_metadynamics_dipeptide_simulations(
        input_pdb=args.input,
        output_dir=args.output,
        steps=args.steps,
        temperature=args.temperature,
        timestep=args.timestep,
        frequency=args.frequency,
        system_type=args.system,
        sigma=args.sigma,
        forcefield_type=args.forcefield,
        charge=args.charge,
        model_path=args.model_path
    )
