import numpy as np
from ase.atoms import Atoms
import copy
from catlearn.optimize.io import array_to_ase, array_to_atoms


def get_energy_catlearn(self, x=None, magmoms=None):

    """ Evaluates the objective function at a given point in space.

    Parameters
    ----------
    self: arrays
        Previous information from the CatLearn optimizer.
    x : array
        Array containing the atomic positions (flatten) or point in space.
    magmoms : array
        List containing the magnetic moments of each Atom.

    Returns
    -------
    energy : float
        The function evaluation value.
    """
    energy = 0.0

    # If no point is passed, evaluate the last trained point.
    if x is None:
        x = self.list_train[-1]

    # Get energies using ASE:
    if self.ase:
        pos_ase = array_to_ase(x, self.num_atoms)
        if magmoms is None:
            self.ase_ini.set_calculator(None)
            self.ase_ini = Atoms(self.ase_ini, positions=pos_ase,
                                 calculator=copy.deepcopy(self.ase_calc))
        if magmoms is not None:
            self.ase_ini.set_calculator(None)
            self.ase_ini = Atoms(self.ase_ini, positions=pos_ase,
                                 calculator=copy.deepcopy(self.ase_calc),
                                 magmoms=magmoms)
        energy = self.ase_ini.get_potential_energy()
        print('Energy of the geometry evaluated (eV):', energy)

    # When not using ASE:
    if not self.ase:
        energy = self.fun.evaluate(x)
        print('Function evaluation:', energy)
    return energy


def get_forces_catlearn(self, x=None):

    """ Evaluates the forces (ASE) or the Jacobian of the objective
    function at a given point in space.

    Parameters
    ----------
    self: arrays
        Previous information from the CatLearn optimizer.
    x : array
        Atoms positions or point in space.

    Returns
    -------
    forces : array
        Forces of the atomic structure (flatten) or the negative value of the
        Jacobian for non atomistic functions.
    """
    forces = 0.0
    # If no point is passed, evaluate the last trained point.
    if x is None:
        x = self.list_train[-1]

    # Get energies using ASE:
    if self.ase:
        forces = self.ase_ini.get_forces().flatten()
        print("Forces of the geometry evaluated (eV/Angst):\n",
              array_to_atoms(forces))

    # When not using ASE:
    if not self.ase:
        forces = -self.fun.jacobian(x)
        print("\nForces:\n", forces)
    return forces


def eval_and_append(self, interesting_point, interesting_magmom=None):
    """ Evaluates the energy and forces (ASE) of the point of interest
        for a given atomistic structure.

    Parameters
    ----------
    self: arrays
        Previous information from the CatLearn optimizer.
    interesting_point : ndarray
        Atoms positions or point in space.
    interesting_magmom: ndarray
        Guessed magnetic moments for the interesting point.

    Return
    -------
    Append function evaluation and forces values to the training set.
    """

    if np.ndim(interesting_point) == 1:
        interesting_point = np.array([interesting_point])

    self.list_train = np.append(self.list_train,
                                interesting_point, axis=0)

    energy = get_energy_catlearn(self, magmoms=interesting_magmom)

    self.list_targets = np.append(self.list_targets, energy)

    gradients = [-get_forces_catlearn(self).flatten()]
    self.list_gradients = np.append(self.list_gradients,
                                    gradients, axis=0)

    self.list_targets = np.reshape(self.list_targets,
                                   (len(self.list_targets), 1))

    self.iter = self.iter + 1
    self.feval += 1
