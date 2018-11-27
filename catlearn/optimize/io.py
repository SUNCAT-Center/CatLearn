from catlearn.optimize.convergence import converged
from prettytable import PrettyTable
import numpy as np
from ase.io import Trajectory, write
import pandas as pd
import datetime
from ase.io.trajectory import TrajectoryWriter


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
    self.table_results = PrettyTable(['Method', 'Step', 'Time', 'Energy',
                                      'fmax', 'Converged?'])


def print_info(self):
    """ Prints the information of the surrogate model convergence at each step.
    """

    if self.iter == 0 and self.feval == 1:
        _start_table(self)

    if self.iter == 0 and self.feval > 1:
        if self.start_mode is 'trajectory':
            _start_table(self)
            for i in range(0, len(self.list_targets)):
                self.table_results.add_row(['Trajectory', 0, print_time(),
                                           self.list_targets[i][0],
                                           self.list_max_abs_forces[i],
                                           converged(self)])
    if self.iter == 0:
        if self.feval == 1:
            self.table_results.add_row(['Eval.', self.iter,
                                        print_time(),
                                        self.list_targets[-1][0],
                                        self.max_abs_forces,
                                        converged(self)])

        if self.feval == 2:
            if self.jac:
                self.table_results.add_row(['LineSearch', self.iter,
                                            print_time(),
                                            self.list_targets[-1][0],
                                            self.max_abs_forces,
                                            converged(self)])
    if self.iter > 0:
        self.table_results.add_row(['MLMin',
                                    self.iter,
                                    print_time(),
                                    self.list_targets[-1][0],
                                    self.max_abs_forces, converged(self)])
    print(self.table_results)


def store_results_neb(self):
    """ Function that print in csv files the predicted NEB curves after
        each iteration"""

    # Save discrete path:
    data = {'Path distance (Angstrom)': self.s,
            'Energy (eV)': self.e,
            'Uncertainty (eV)': self.uncertainty_path}
    df = pd.DataFrame(data,
                      columns=['Path distance (Angstrom)', 'Energy (eV)',
                               'Uncertainty (eV)'])
    df.to_csv('results_neb.csv')

    # Save interpolated path:
    data = {'Path distance (Angstrom)': self.sfit, 'Energy (eV)': self.efit}

    df = pd.DataFrame(data,
                      columns=['Path distance (Angstrom)', 'Energy (eV)'])
    df.to_csv('results_neb_interpolation.csv')


def store_trajectory_neb(self):
    # Evaluated images.
    TrajectoryWriter(atoms=self.ase_ini,
                     filename='./evaluated_structures.traj',
                     mode='a').write()
    # Last path.
    write('last_predicted_path.traj', self.images)

    # All paths.
    for i in self.images:
        TrajectoryWriter(atoms=i,
                         filename='all_predicted_paths.traj',
                         mode='a').write()


def print_time():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


def print_version(version):
    print(""" 
       ____      _   _                          
      / ___|__ _| |_| |    ___  __ _ _ __ _ __  
     | |   / _` | __| |   / _ \/ _` | '__| '_ \ 
     | |__| (_| | |_| |__|  __/ (_| | |  | | | |
      \____\__,_|\__|_____\___|\__,_|_|  |_| |_| """ + version + """
      
      """)
