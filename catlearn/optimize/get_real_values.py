import numpy as np
from ase.atoms import Atoms
from catlearn.optimize.io import array_to_ase


def get_energy_catlearn(self, x=None):

    """ Evaluates the objective function at a given point in space.

    Parameters
    ----------
    self: arrays
        Previous information from the CatLearn optimizer.
    x : array
        Array containing the atomic positions (flatten).

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
    pos_ase = array_to_ase(x, self.num_atoms)

    self.ase_ini.set_calculator(None)
    self.ase_ini = Atoms(self.ase_ini, positions=pos_ase,
                         calculator=self.ase_calc)
    energy = self.ase_ini.get_potential_energy()
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
        Forces of the atomic structure (flatten).
    """
    forces = 0.0
    # If no point is passed, evaluate the last trained point.
    if x is None:
        x = self.list_train[-1]

    # Get energies using ASE:
    forces = self.ase_ini.get_forces().flatten()
    return forces


def eval_and_append(self, interesting_point):
    """ Evaluates the energy and forces (ASE) of the point of interest
        for a given atomistic structure.

    Parameters
    ----------
    self: arrays
        Previous information from the CatLearn optimizer.
    interesting_point : ndarray
        Atoms positions or point in space.

    Return
    -------
    Append function evaluation and forces values to the training set.
    """

    if np.ndim(interesting_point) == 1:
        interesting_point = np.array([interesting_point])

    self.list_train = np.append(self.list_train,
                                interesting_point, axis=0)

    energy = get_energy_catlearn(self)

    self.list_targets = np.append(self.list_targets, energy)

    gradients = [-get_forces_catlearn(self).flatten()]
    self.list_gradients = np.append(self.list_gradients,
                                    gradients, axis=0)

    self.list_targets = np.reshape(self.list_targets,
                                   (len(self.list_targets), 1))

    self.iter = self.iter + 1
    self.feval += 1
