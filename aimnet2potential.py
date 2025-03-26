"""
aimnet2potential.py: Implements the AIMNet2 potential function.

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2025 xAI and the Authors.
Authors: [Your Name]
Contributors:

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import openmm
from openmmml.mlpotential import MLPotentialImpl, MLPotentialImplFactory
from typing import Iterable, Optional

class AIMNet2PotentialImplFactory(MLPotentialImplFactory):
    """Factory that creates AIMNet2PotentialImpl objects."""

    def createImpl(self, name: str, modelPath: str, charge: float = 0.0, **args) -> MLPotentialImpl:
        """
        Create an AIMNet2PotentialImpl instance.

        Parameters
        ----------
        name : str
            The name of the potential (e.g., 'aimnet2').
        modelPath : str
            Path to the AIMNet2 model file (e.g., 'aimnet2_b973c_ens.jpt').
        charge : float
            The total charge of the system (default 0.0).
        args : dict
            Additional arguments for customization.
        """
        return AIMNet2PotentialImpl(name, modelPath, charge)

class AIMNet2PotentialImpl(MLPotentialImpl):
    """Implementation of the AIMNet2 potential for OpenMM.

    This class integrates an AIMNet2 model into OpenMM using TorchForce. The model
    is loaded from a specified path and used to compute energies and forces.

    Example usage:
    >>> potential = MLPotential('aimnet2', modelPath='aimnet2_b973c_ens.jpt', charge=0.0)
    >>> system = potential.createSystem(topology)
    """

    def __init__(self, name: str, modelPath: str, charge: float):
        """
        Initialize the AIMNet2PotentialImpl.

        Parameters
        ----------
        name : str
            The name of the potential (e.g., 'aimnet2').
        modelPath : str
            Path to the AIMNet2 model file.
        charge : float
            The total charge of the system.
        """
        self.name = name
        self.modelPath = modelPath
        self.charge = charge

    def addForces(
        self,
        topology: openmm.app.Topology,
        system: openmm.System,
        atoms: Optional[Iterable[int]],
        forceGroup: int,
        precision: Optional[str] = None,
        **args
    ) -> None:
        """
        Add the AIMNet2 force to the OpenMM System.

        Parameters
        ----------
        topology : openmm.app.Topology
            The topology of the system.
        system : openmm.System
            The system to which the force will be added.
        atoms : Iterable[int], optional
            Indices of atoms to include in the model. If None, all atoms are included.
        forceGroup : int
            The force group for the force.
        precision : str, optional
            Precision for the model ('single' or 'double'). Defaults to model's precision.
        """
        import torch
        import openmmtorch

        # Load the AIMNet2 model
        device = 'cpu' 
        model = torch.jit.load(self.modelPath, map_location=device)

        # Determine precision
        model_default_dtype = next(model.parameters()).dtype
        if precision is None:
            dtype = model_default_dtype
        elif precision == 'single':
            dtype = torch.float32
        elif precision == 'double':
            dtype = torch.float64
        else:
            raise ValueError("Supported precision values are 'single' or 'double'.")
        if dtype != model_default_dtype:
            print(f"Converting model from {model_default_dtype} to {dtype}.")

        # Get atomic numbers from topology
        included_atoms = list(topology.atoms())
        if atoms is not None:
            included_atoms = [included_atoms[i] for i in atoms]
        numbers = torch.tensor([atom.element.atomic_number for atom in included_atoms], dtype=torch.long)

        class AIMNet2Force(torch.nn.Module):
            """PyTorch module wrapping the AIMNet2 model for OpenMM."""

            def __init__(
                self,
                model: torch.jit._script.RecursiveScriptModule,
                numbers: torch.Tensor,
                charge: float,
                atoms: Optional[Iterable[int]],
                dtype: torch.dtype
            ):
                super(AIMNet2Force, self).__init__()
                self.model = model.to(dtype)
                self.numbers = torch.nn.Parameter(numbers.unsqueeze(0), requires_grad=False)
                self.charge = torch.nn.Parameter(torch.tensor([charge], dtype=dtype), requires_grad=False)
                self.dtype = dtype
                self.hartree2kjmol = 2625.499638  # Hartree to kJ/mol
                if atoms is None:
                    self.indices = None
                else:
                    self.indices = torch.tensor(sorted(atoms), dtype=torch.int64)

            def forward(self, positions: torch.Tensor, boxvectors: Optional[torch.Tensor] = None) -> torch.Tensor:
                """
                Compute the energy using the AIMNet2 model.

                Parameters
                ----------
                positions : torch.Tensor
                    Atomic positions in nm.
                boxvectors : torch.Tensor, optional
                    Periodic box vectors in nm.

                Returns
                -------
                energy : torch.Tensor
                    Energy in kJ/mol.
                """
                positions = positions.to(self.dtype)
                if self.indices is not None:
                    positions = positions[self.indices]

                # AIMNet2 expects coordinates in Angstroms
                coord = positions * 10.0  # nm to Angstrom
                coord.requires_grad_(True)

                # Prepare input dictionary
                input_dict = {
                    'coord': coord.unsqueeze(0),  # Add batch dimension
                    'numbers': self.numbers,
                    'charge': self.charge
                }

                # Compute energy in eV
                out = self.model(input_dict)
                energy_ev = out['energy']

                # Convert to kJ/mol (eV -> Hartree -> kJ/mol)
                energy_hartree = energy_ev * (1 / 27.211386245988)  # eV to Hartree
                energy_kjmol = energy_hartree * self.hartree2kjmol

                return energy_kjmol

        # Create the force object
        aimnet2_force = AIMNet2Force(model, numbers, self.charge, atoms, dtype)

        # Convert to TorchScript
        module = torch.jit.script(aimnet2_force)

        # Add to OpenMM system via TorchForce
        force = openmmtorch.TorchForce(module)
        force.setForceGroup(forceGroup)
        is_periodic = (topology.getPeriodicBoxVectors() is not None) or system.usesPeriodicBoundaryConditions()
        force.setUsesPeriodicBoundaryConditions(is_periodic)
        system.addForce(force)

# Register the factory (place this in your script or module initialization)
