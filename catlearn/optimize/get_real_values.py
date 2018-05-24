import numpy as np
from catlearn.optimize.convert import *
from ase.atoms import Atoms
from ase.visualize import view

def get_energy_catlearn(self, x=None):

    """ Evaluates the objective function at a given point in space.

    Parameters
    ----------
    x : array
        Array containing the atom positions (flatten) or point in space.

    Returns
    -------
    energy : float
        The function evaluation value.
    """

    # If no point is passed, evaluate the last trained point.

    if x is None:
        x = self.list_train[-1]

    # ASE:
    if self.ase:
        pos_ase = array_to_ase(x, self.num_atoms)
        self.ase_ini = Atoms(self.ase_ini, positions=pos_ase,
            calculator=self.ase_calc)
        energy = self.ase_ini.get_potential_energy()
        # print('Energy (eV):', energy)

    # When not using ASE:
    if not self.ase:
        energy = self.fun.evaluate(x)
        # print('Function evaluation:', energy)
    return energy

def get_forces_catlearn(self, x=None):

    """ Evaluates the forces (ASE) or the Jacobian of the objective
    function at a given point in given space.

    Parameters
    ----------
    x : array
        Atoms positions or point in a given space.

    Returns
    -------
    forces : array
        Forces (ASE) or the negative value of the Jacobian (for other
        functions).
    """

    # If no point is passed, evaluate the last trained point.
    if x is None:
        x = self.list_train[-1]

    # ASE:
    if self.ase:
        pos_ase = array_to_ase(x, self.num_atoms)
        forces = self.ase_ini.get_forces().flatten()
        # print('Forces (eV/Angst):\n', forces)

    # When not using ASE:
    if not self.ase:
        forces = -self.fun.jacobian(x)
        # print('Forces:\n', forces)
    return forces