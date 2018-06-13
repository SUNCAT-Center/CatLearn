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

    #  The force on all individual atoms should be less than fmax.
    if self.ase is True:
        if self.jac is True:
            self.list_fmax = get_fmax(self.list_gradients, self.num_atoms)
            self.max_abs_forces = self.list_fmax[-1][0]
            if self.min_iter:
                if self.iter <= self.min_iter:
                    return False
            if self.max_abs_forces < self.fmax:
                return True

    # The force on all individual components should be less than fmax.
    if self.ase is False:
        if self.jac is True:
            self.list_fmax = np.amax(np.abs(self.list_gradients), axis=1)
            forces_last_iteration = -self.list_fmax[-1]
            self.max_abs_forces = np.max(np.abs(forces_last_iteration))
            if self.min_iter:
                if self.iter <= self.min_iter:
                    return False
            if self.max_abs_forces < self.fmax:
                return True

    # Check energy convergence.
    if self.feval > 1:
        self.e_diff = np.abs(self.list_targets[-1] - self.list_targets[-2])

    if self.min_iter:
        if self.iter <= self.min_iter:
            return False

    if not self.jac:
        if len(self.list_targets) > 1:
            if self.e_diff < self.e_max:
                return True

    return False
