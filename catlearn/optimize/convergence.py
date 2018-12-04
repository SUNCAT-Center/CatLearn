import numpy as np


def get_fmax(gradients_flatten, num_atoms):
    """Function that print a list of max. individual atom forces."""

    list_fmax = np.zeros((len(gradients_flatten), 1))
    j = 0
    for i in gradients_flatten:
        atoms_forces_i = np.reshape(i, (num_atoms, 3))
        list_fmax[j] = np.max(np.sqrt(np.sum(atoms_forces_i**2, axis=1)))
        j = j + 1
    return list_fmax


def converged(self):
    """Function that checks the convergence in each optimization step."""

    self.list_fmax = get_fmax(-np.array([self.list_gradients[-1]]),
                              self.num_atoms)
    self.max_abs_forces = np.max(np.abs(self.list_fmax))

    if self.max_abs_forces < self.fmax:
        print('Congratulations. Optimization converged.')
        print('All the evaluated structures can be found in:',
              self.filename)
        return True

    return False


def converged_dimer(self):
    """Function that checks the convergence in each optimization step."""
    if len(self.list_targets) > 1:
        self.list_fmax = get_fmax(-np.array([self.list_gradients[-1]]),
                                  self.num_atoms)
        self.max_abs_forces = np.max(np.abs(self.list_fmax))
        if self.max_abs_forces < self.fmax:
            print('Congratulations. Dimer optimization has converged.')
            print('All the evaluated structures can be found in:',
                  self.filename)
            return True
    return False
