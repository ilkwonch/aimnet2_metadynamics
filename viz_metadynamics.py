import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import mdtraj as md
import seaborn as sns
from matplotlib.colors import LogNorm
import argparse
import pandas as pd
import glob
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 10
rcParams["figure.figsize"] = [3, 3]
rcParams["figure.dpi"] = 300.0


def plot_ramachandran(trajectory_file, topology_file, output_dir):
    """
    Generate a Ramachandran plot from the trajectory data
    """
    # Load trajectory
    print(f"Loading trajectory: {trajectory_file}")
    traj = md.load(trajectory_file, top=topology_file)
    
    # Calculate phi and psi angles
    print("Calculating phi and psi angles...")
    phi = md.compute_phi(traj)[1] * 180.0 / np.pi  # Convert to degrees
    psi = md.compute_psi(traj)[1] * 180.0 / np.pi  # Convert to degrees

    # Create 2D histogram
    print("Creating Ramachandran plot...")
    plt.figure(figsize=(10, 8))
    
    # Create a 2D histogram
    hist, xedges, yedges = np.histogram2d(phi.flatten(), psi.flatten(), 
                                         bins=72, range=[[-180, 180], [-180, 180]])
    
    # Normalize and log-scale for better visualization
    hist = hist / np.sum(hist)
    hist = np.ma.masked_where(hist == 0, hist)
    
    # Plot as a pcolormesh
    plt.pcolormesh(xedges, yedges, hist.T, cmap='viridis', norm=LogNorm())
    
    #plt.pcolormesh(xedges, yedges, hist.T, cmap='viridis')
    
    #levels = np.linspace(0.001, np.max(hist), 5)  # Linearly spaced levels
    #contour = plt.contour(0.5 * (xedges[1:] + xedges[:-1]),
    #                      0.5 * (yedges[1:] + yedges[:-1]),
    #                      hist.T, levels=levels, colors='white', alpha=0.5)

    #plt.colorbar(label='Probability (log scale)')
    
    # Add contour lines for visibility
    levels = np.logspace(-5, -1, 5)
    contour = plt.contour(0.5 * (xedges[1:] + xedges[:-1]), 
                          0.5 * (yedges[1:] + yedges[:-1]), 
                          hist.T, levels=levels, colors='white', alpha=0.5)
    
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$\psi$')
    #plt.title('Ramachandran')
    plt.grid(linestyle='--', alpha=0.3)
    
    # Mark the key regions
    regions = {
        #'α-helix (left-handed)': (60, 40),
        'α-helix (right-handed)': (-60, -40),
        'β-sheet': (-120, 120),
        'left-handed α': (60, 30)
    }
    
    for name, (phi_pos, psi_pos) in regions.items():
        plt.annotate(name, xy=(phi_pos, psi_pos), xycoords='data',
                    fontsize=10, color='white',
                    xytext=(0, 0), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ramachandran_plot.png'), dpi=300)
    plt.close()
    print(f"Ramachandran plot saved to {os.path.join(output_dir, 'ramachandran_plot.png')}")

def plot_dihedral_time_series(state_file, output_dir,setup_temp=300,free_energy_file=None):
    """
    Plot the time evolution of the phi and psi angles and potential energy
    """
    print(f"Loading state data: {state_file}")
    
    # Load state data
    state_data = pd.read_csv(state_file)
    # Extract relevant columns - this might need adjustment based on your actual CSV format
    #state_data['#"Step"'] = state_data['#"Step"'] / 500
    
    state_data['#"Step"'] = state_data['#"Step"'] / 1000
    
    steps = state_data['#"Step"']
    kj_to_kcal = 1 / 4.184
    state_data['Potential Energy (kcal/mol)'] =state_data['Potential Energy (kJ/mole)'] * kj_to_kcal
    potential_energy = state_data['Potential Energy (kcal/mol)']
    temperature = state_data['Temperature (K)']
    
    # Plot potential energy vs time
    

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=state_data, x='#"Step"', y='Potential Energy (kcal/mol)', linewidth=2)
    #plt.plot(steps, potential_energy, 'b-', linewidth=1.5)
    plt.xlabel('Time (ps)')
    plt.ylabel('Potential Energy (kcal/mol)')
    #plt.title('Potential Energy Evolution During Metadynamics')
    plt.grid(linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'potential_energy.png'), dpi=300)
    plt.close()
    
    # Plot temperature vs time
    plt.figure(figsize=(12, 6))
    plt.plot(steps, temperature, 'r-', linewidth=1.5)
    plt.xlabel('Time (ps)')
    plt.ylabel('Temperature (K)')
    #plt.title('Temperature Evolution During Metadynamics')
    plt.grid(linestyle='--', alpha=0.3)
    #plt.axhline(y=300, color='k', linestyle='--', alpha=0.5, label='Target Temperature (300K)')
    #plt.axhline(y=temperature, color='k', linestyle='--', alpha=0.5, label=f'Target Temperature ({setup_temp}K)')
    plt.axhline(y=setup_temp, color='k', linestyle='--', alpha=0.5, label=f'Target Temperature ({setup_temp}K)')

    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temperature.png'), dpi=300)
    plt.close()
    
    print(f"Energy and temperature plots saved to {output_dir}")
    
    # If free energy data is available, plot convergence
    if free_energy_file and os.path.exists(free_energy_file):
        plot_free_energy_convergence(free_energy_file, output_dir)

def plot_free_energy_convergence(free_energy_file, output_dir):
    """
    Plot the free energy convergence from the free energy file
    """
    print(f"Loading free energy data: {free_energy_file}")
    
    # Load free energy data
    free_energy = np.loadtxt(free_energy_file)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(free_energy, origin='lower', cmap='viridis', aspect='auto', 
              extent=[-180, 180, -180, 180])
    plt.colorbar(label="Free Energy (kJ/mol)")
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$\psi$')
    plt.title('Free Energy Landscape')
    
    # Mark the key regions
    regions = {
        'α-helix': (-60, -40),
        'β-sheet': (-120, 120),
        'left-handed α': (60, 30)
    }
    
    for name, (phi_pos, psi_pos) in regions.items():
        plt.annotate(name, xy=(phi_pos, psi_pos), xycoords='data',
                    fontsize=10, color='white',
                    xytext=(0, 0), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'free_energy_dihedrals.png'), dpi=300)
    plt.close()
    print(f"Annotated free energy plot saved to {os.path.join(output_dir, 'free_energy_annotated.png')}")




def create_3d_energy_landscape(free_energy_file, output_dir):
    """
    Create a 3D visualization of the free energy landscape
    """
    print(f"Creating 3D energy landscape from: {free_energy_file}")
    
    # Load free energy data
    free_energy = np.loadtxt(free_energy_file)
    
    # Create a 3D plot
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a mesh grid for phi and psi
    phi = np.linspace(-180, 180, free_energy.shape[0])
    psi = np.linspace(-180, 180, free_energy.shape[1])
    phi_mesh, psi_mesh = np.meshgrid(phi, psi)
    
    # Clip extreme values for better visualization
    max_energy = np.percentile(free_energy, 95)
    free_energy_clipped = np.clip(free_energy, None, max_energy)
    
    # Plot the surface
    surf = ax.plot_surface(phi_mesh, psi_mesh, free_energy_clipped.T, 
                          cmap='viridis', edgecolor='none', alpha=0.8)
    
    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Free Energy (kJ/mol)')
    
    # Add contour projection on xy-plane for clarity
    offset = np.min(free_energy_clipped) - 5
    contour = ax.contour(phi_mesh, psi_mesh, free_energy_clipped.T, 
                        zdir='z', offset=offset, cmap='viridis', linewidths=2)
    
    ax.set_xlabel(r'$\phi$ (degrees)')
    ax.set_ylabel(r'$\psi$ (degrees)')
    ax.set_zlabel('Free Energy (kJ/mol)')
    ax.set_title('3D Free Energy Landscape')
    
    # Set viewing angle
    ax.view_init(elev=30, azim=-45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'free_energy_3d.png'), dpi=300)
    plt.close()
    print(f"3D energy landscape saved to {os.path.join(output_dir, 'free_energy_3d.png')}")

def main():
    parser = argparse.ArgumentParser(description="Visualize metadynamics simulation results")
    parser.add_argument('-d', '--directory', type=str, required=True, help='Simulation output directory')
    parser.add_argument('-p', '--pdb', type=str, required=True, help='Input PDB file used for simulation')
    parser.add_argument('-t', '--temperature',  type=int, required=True, help='Temperature')
    args = parser.parse_args()
    
    simulation_dir = args.directory
    pdb_file = args.pdb
    setup_temp  = args.temperature
    energy_dir = os.path.join(simulation_dir, 'energy')
    
    # Create a directory for visualizations
    viz_dir = os.path.join(simulation_dir, 'visualizations')
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    
    # Check for required files
    trajectory_file = os.path.join(energy_dir, 'trajectory.dcd')
    state_file = os.path.join(simulation_dir, 'state.csv')
    free_energy_file = os.path.join(simulation_dir, 'free_energy.txt')
    
    # Verify files exist
    if not os.path.exists(trajectory_file):
        print(f"Warning: Trajectory file not found at {trajectory_file}")
    else:
        plot_ramachandran(trajectory_file, pdb_file, viz_dir)
    if not os.path.exists(state_file):
        print(f"Warning: State file not found at {state_file}")
    else:
        plot_dihedral_time_series(state_file, viz_dir, setup_temp,free_energy_file)
    
    if not os.path.exists(free_energy_file):
        print(f"Warning: Free energy file not found at {free_energy_file}")
    else:
        create_3d_energy_landscape(free_energy_file, viz_dir)
    
    # Analyze metadynamics bias files if they exist
    #analyze_metadynamics_bias(energy_dir, viz_dir)
    
    print(f"\nAll visualizations have been saved to {viz_dir}")
    print("To view the results, you can open the PNG files in any image viewer.")

if __name__ == "__main__":
    main()
