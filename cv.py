import os
import mdtraj as md
from openmm.openmm import CustomBondForce, CustomAngleForce, CustomTorsionForce

def get_phi_psi_indices(pdb_file):
    traj = md.load(pdb_file)
    
    
    phi_indices, _ = md.compute_phi(traj)  # Shape: (n_frames, n_phi, 4)
    psi_indices, _ = md.compute_psi(traj)
    #phi_indices = md.compute_phi(traj)[0][0]  # Shape: (n_frames, n_phi, 4)
    #psi_indices = md.compute_psi(traj)[0][0]  
    
    if len(phi_indices) > 0:
        phi_atoms = phi_indices[0]  # Take the first phi dihedral
    else:
        raise ValueError("No phi dihedral found in the topology.")
    
    if len(psi_indices) > 0:
        psi_atoms = psi_indices[0]  # Take the first psi dihedral
    else:
        raise ValueError("No psi dihedral found in the topology.")
    
    return phi_atoms, psi_atoms

def get_distance_cv(pdb_file, atom_pair):
    traj = md.load(pdb_file)
    n_atoms = traj.n_atoms
    
    if not (0 <= atom_pair[0] < n_atoms and 0 <= atom_pair[1] < n_atoms):
        raise ValueError(f"Atom indices {atom_pair} are out of range for {n_atoms} atoms.")
    
    distance_cv = CustomBondForce("r")
    distance_cv.addBond(int(atom_pair[0]), int(atom_pair[1]), [])
    return distance_cv

def get_angle_cv(pdb_file, atom_triple):
    traj = md.load(pdb_file)
    n_atoms = traj.n_atoms
    
    for idx in atom_triple:
        if not (0 <= idx < n_atoms):
            raise ValueError(f"Atom index {idx} is out of range for {n_atoms} atoms.")
    
    angle_cv = CustomAngleForce("theta")
    angle_cv.addAngle(int(atom_triple[0]), int(atom_triple[1]), int(atom_triple[2]), [])
    return angle_cv

def get_dihedral_cv(pdb_file, atom_quad):
    
    traj = md.load(pdb_file)
    n_atoms = traj.n_atoms
    
    for idx in atom_quad:
        if not (0 <= idx < n_atoms):
            raise ValueError(f"Atom index {idx} is out of range for {n_atoms} atoms.")
    
    dihedral_cv = CustomTorsionForce("theta")
    dihedral_cv.addTorsion(int(atom_quad[0]), int(atom_quad[1]), int(atom_quad[2]), int(atom_quad[3]), [])
    return dihedral_cv

