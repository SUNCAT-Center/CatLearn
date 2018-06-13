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
            print('Spin polarized calculation.')
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
        print('Forces of the geometry evaluated (eV/Angst):\n',
              array_to_atoms(forces))

    # When not using ASE:
    if not self.ase:
        forces = -self.fun.jacobian(x)
        print('Forces:\n', forces)
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
    energy_1 = get_energy_catlearn(self, magmoms=self.magmom_is)

    self.list_targets = np.append(self.list_targets, energy_1)

    gradients_1 = [-get_forces_catlearn(self).flatten()]
    self.list_gradients = np.append(self.list_gradients,
                                    gradients_1, axis=0)

    if self.spin is True:
        if np.ndim(interesting_point) == 1:
            interesting_point = np.array([interesting_point])

        self.list_train = np.append(self.list_train,
                                    interesting_point, axis=0)
        energy_2 = get_energy_catlearn(self, magmoms=self.magmom_fs)

        self.list_targets = np.append(self.list_targets, energy_2)
        self.list_targets = np.reshape(self.list_targets,
                                       (len(self.list_targets), 1))

        gradients_2 = [-get_forces_catlearn(self).flatten()]
        self.list_gradients = np.append(self.list_gradients,
                                        gradients_2, axis=0)

        last_index = len(self.list_targets)-1


        if energy_1 >= energy_2:
            self.list_train = np.delete(self.list_train, last_index, axis=0)
            self.list_targets = np.delete(self.list_targets, last_index,
                                          axis=0)
            self.list_gradients = np.delete(self.list_gradients, last_index,
                                            axis=0)

        if energy_1 < energy_2:
            self.list_train = np.delete(self.list_train, last_index-1,
                                        axis=0)
            self.list_targets = np.delete(self.list_targets, last_index-1,
                                          axis=0)
            self.list_gradients = np.delete(self.list_gradients,
                                            last_index-1, axis=0)

    self.list_targets = np.reshape(self.list_targets,
                                   (len(self.list_targets), 1))

    self.iter += 1
