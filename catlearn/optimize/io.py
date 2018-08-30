from catlearn.optimize.convergence import converged
from prettytable import PrettyTable
import os
import numpy as np
from ase.io import Trajectory
import pandas as pd


def ase_traj_to_catlearn(traj_file):
    """Converts a trajectory file from ASE to a list of train, targets and
    gradients. The first and last images of the trajectory file are also
    included in this dictionary.

        Parameters
        ----------
        traj_file : string
            Name of the trajectory file to open. The file must be in
            the current working directory.

        Returns
        -----------
        results: dict
            Dictionary that contains list of train (including constraints),
            targets and gradients, number of atoms for the atomistic
            structure, images included in the trajectory file and
            Atoms structures of the initial and final endpoints of the NEB.

    """
    traj_ase = Trajectory(traj_file)
    first_image = traj_ase[0]
    images = []
    constraints = first_image._get_constraints()
    number_images = len(traj_ase)
    num_atoms = first_image.get_number_of_atoms()
    list_train = []
    list_targets = []
    list_gradients = []
    for i in range(0, number_images):
        image_i = traj_ase[i]
        images.append(image_i)
        list_train = np.append(list_train,
                               image_i.get_positions().flatten(), axis=0)
        list_targets = np.append(list_targets, image_i.get_potential_energy())
        list_gradients = np.append(list_gradients,
                                   -(image_i.get_forces().flatten()), axis=0)
    list_train = np.reshape(list_train, (number_images, num_atoms * 3))
    list_targets = np.reshape(list_targets, (number_images, 1))
    list_gradients = np.reshape(list_gradients, (number_images, num_atoms * 3))
    results = {'first_image': images[0],
               'last_image': images[1],
               'images': images,
               'list_train': list_train,
               'list_targets': list_targets,
               'list_gradients': list_gradients,
               'constraints': constraints,
               'num_atoms': num_atoms}
    return results


def array_to_ase(input_array, num_atoms):
    """Converts a flat array into an ase structure (list).

        Parameters
        ----------
        input_array : ndarray
            Structure.
        num_atoms : int
            Number of atoms.

        Returns
        -----------
        pos_ase: list
            Position of the atoms in ASE format.

    """
    atoms_pos = np.reshape(input_array, (num_atoms, 3))
    x_pos = atoms_pos[:, 0]
    y_pos = atoms_pos[:, 1]
    z_pos = atoms_pos[:, 2]
    pos_ase = list(zip(x_pos, y_pos, z_pos))
    return pos_ase


def array_to_atoms(input_array):
    """ Converts an input flat array into atoms shape for ASE.

    Parameters
    ----------
    input_array : ndarray
        Structure.

    Returns
    -----------
    pos_ase: list
    Position of the atoms in ASE format.
    """

    atoms = np.reshape(input_array, (int(len(input_array) / 3), 3))  # Check.
    return atoms


def _start_table(self):
        if not self.jac:
            self.table_results = PrettyTable(['Method', 'Iterations',
                                              'Func. evaluations',
                                              'Function value',
                                              'e_diff',
                                              'Converged?'])
        if self.jac:
            self.table_results = PrettyTable(['Method', 'Iterations',
                                              'Func. evaluations', 'Function '
                                              'value', 'fmax', 'Converged?'])


def print_info(self):
    """ Prints the information of the surrogate model convergence at each step.
    """

    if self.iter == 0 and self.feval == 1:
        _start_table(self)

    if self.iter == 1 and self.feval > 1:
        if self.start_mode is 'dict':
            _start_table(self)
            for i in range(0, self.feval-1):
                if self.jac:
                    self.table_results.add_row(['Previous', self.iter-1,
                                               i+1, self.list_targets[i][0],
                                               self.list_fmax[i],
                                               converged(self)])
                if not self.jac:
                    diff_energ = ['-']
                    if i != 0:
                        diff_energ = self.list_targets[i-1] \
                                     - self.list_targets[i]
                    self.table_results.add_row(['Previous', self.iter-1,
                                                i+1, self.list_targets[i][0],
                                                diff_energ, converged(self)])

        if self.start_mode is 'trajectory':
            _start_table(self)
            for i in range(0, self.feval-1):
                self.table_results.add_row(['Traj. ASE', self.iter-1, i+1,
                                           self.list_targets[i][0],
                                           self.list_fmax[i][0],
                                           converged(self)])

    if self.iter == 0:
        if self.feval == 1:
            if not self.jac:
                self.table_results.add_row(['Eval.', self.iter,
                                           self.feval,
                                           self.list_targets[-1][0], '-',
                                           converged(self)])
            if self.jac:
                self.table_results.add_row(['Eval.', self.iter,
                                            self.feval,
                                            self.list_targets[-1][0],
                                            self.max_abs_forces,
                                            converged(self)])

        if self.feval == 2:
            if not self.i_ase_step:
                if not self.jac:
                    self.table_results.add_row(['LineSearch', self.iter,
                                                self.feval,
                                                self.list_targets[-1][0], '-',
                                                converged(self)])
                if self.jac:
                    self.table_results.add_row(['LineSearch', self.iter,
                                                self.feval,
                                                self.list_targets[-1][0],
                                                self.max_abs_forces,
                                                converged(self)])
            if self.i_ase_step:
                if not self.jac:
                    self.table_results.add_row([self.i_ase_step, self.iter,
                                                self.feval,
                                                self.list_targets[-1][0],
                                                self.e_diff, converged(self)])
                if self.jac:
                    self.table_results.add_row([self.i_ase_step, self.iter,
                                                self.feval,
                                                self.list_targets[-1][0],
                                                self.max_abs_forces,
                                                converged(self)])

    if self.iter > 0:
        if not self.jac:
            self.table_results.add_row(['CatLearn', self.iter,
                                        self.feval,
                                        self.list_targets[-1][0],
                                        self.e_diff, converged(self)])
        if self.jac:
            self.table_results.add_row(['CatLearn', self.iter,
                                        self.feval,
                                        self.list_targets[-1][0],
                                        self.max_abs_forces, converged(self)])
    print(self.table_results)
    f_print = open(str(self.filename)+'_convergence_catlearn.txt', 'w')
    f_print.write(str(self.table_results))
    f_print.close()


def print_info_ml(self):
    """ Prints the information of the ML convergence at each step.
    """

    if self.iter == 0:  # Print header
        self.table_ml_results = PrettyTable(['CatLearn iteration', 'Minimizer',
                                             'ML func. evaluations',
                                             'Suggested point (x)',
                                             'Predicted value', 'Converged?'])
    self.table_ml_results.add_row([self.iter, self.ml_algo,
                                   self.ml_feval_pred_mean,
                                   self.interesting_point,
                                   self.ml_f_min_pred_mean,
                                   self.ml_convergence])
    f_ml_print = open(str(self.filename)+'_convergence_ml.txt', 'w')
    f_ml_print.write(str(self.table_ml_results))
    f_ml_print.close()


def store_results(self):
    if not self.jac:
        self.results = {'train': self.list_train,
                        'targets': self.list_targets,
                        'iterations': self.iter,
                        'f_eval': self.feval,
                        'converged': converged(self)}
    if self.jac:
        self.results = {'train': self.list_train,
                        'targets': self.list_targets,
                        'gradients': self.list_gradients,
                        'forces': -self.list_gradients,
                        'iterations': self.iter,
                        'f_eval': self.feval,
                        'converged': converged(self)}

    f_res = open(str(self.filename) + "_data.txt", "w")
    f_res.write(str(self.results))
    f_res.write(str(dict))


def store_results_neb(s, e, sfit, efit, uncertainty_path):
    """ Function that print in csv files the predicted NEB curves after
        each iteration"""

    # Save discrete path:
    data = {'Path distance (Angstrom)': s,
            'Energy (eV)': e,
            'Uncertainty (eV)': uncertainty_path}
    df = pd.DataFrame(data,
                      columns=['Path distance (Angstrom)', 'Energy (eV)',
                               'Uncertainty (eV)'])
    df.to_csv('results_neb.csv')

    # Save interpolated path:
    data = {'Path distance (Angstrom)': sfit, 'Energy (eV)': efit}

    df = pd.DataFrame(data,
                      columns=['Path distance (Angstrom)', 'Energy (eV)'])
    df.to_csv('results_neb_interpolation.csv')
