import numpy as np
from ase.io import write
from ase.parallel import parprint
import datetime
from ase.io.trajectory import TrajectoryWriter


def ase_to_catlearn(list_atoms):
    """Converts a trajectory file from ASE to a list of train, targets and
    gradients. The first and last images of the trajectory file are also
    included in this dictionary.

        Parameters
        ----------
        list_atoms : string
            List Atoms objects in ASE format. The file must be in
            the current working directory.

        Returns
        -----------
        results: dict
            Dictionary that contains list of train (including constraints),
            targets and gradients, number of atoms for the atomistic
            structure, images included in the trajectory file and
            Atoms structures of the initial and final endpoints of the NEB.

    """
    first_image = list_atoms[0]
    images = []
    constraints = first_image.constraints
    number_images = len(list_atoms)
    num_atoms = first_image.get_number_of_atoms()
    list_train = []
    list_targets = []
    list_gradients = []
    for i in range(0, number_images):
        image_i = list_atoms[i]
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


def print_info_neb(self):
    """ Prints the information of the surrogate model convergence at each step.
    """
    if self.iter < 2:
        self.energy_backward = 0.0
        self.energy_forward = 0.0

    iter_tab = self.iter
    ef_tab = np.round(self.energy_forward, 5)
    eb_tab = np.round(self.energy_backward, 5)
    unc_max_tab = np.round(np.max(self.uncertainty_path[1:-1]), 5)
    unc_mean_tab = np.round(np.mean(self.uncertainty_path[1:-1]), 5)
    fmax_tab = np.round(self.max_abs_forces, 6)

    pre_tab_neb = [iter_tab, ef_tab, eb_tab, unc_max_tab, unc_mean_tab,
                   fmax_tab]
    if self.iter == 0:
        self.tab_neb = np.array([pre_tab_neb])
    if self.iter > 0:
        self.tab_neb = np.append(self.tab_neb, [pre_tab_neb], axis=0)

    parprint('+--------+------+---------------------+---------------------+---'
             '------------------+--------------+--------------+----------+')
    parprint('| Method | Step |        Time         | Pred. barrier (-->) | '
             'Pred. barrier (<--) | Max. uncert. | Avg. uncert. |   fmax   |')
    parprint('+--------+------+---------------------+---------------------+---'
             '---------------+--------------+--------------+----------+')
    for i in range(0, self.iter+1):
        parprint('| ML-NEB |'
                 + '{0:6d}|'.format(int(self.tab_neb[i, 0])),
                 print_time()
                 + ' |'
                 + '{0:21f}|'.format(self.tab_neb[i, 1])
                 + '{0:21f}|'.format(self.tab_neb[i, 2])
                 + '{0:14f}|'.format(self.tab_neb[i, 3])
                 + '{0:14f}|'.format(self.tab_neb[i, 4])
                 + '{0:10f}|'.format(self.tab_neb[i, 5]))
    parprint('+--------+------+---------------------+---------------------+---'
             '---------------+--------------+--------------+----------+')


def print_info(self):
    """ Output of the ML-Min surrogate machine learning algorithm.
    """
    energy_tab = np.round(self.list_targets, 7)
    fmax_tab = np.round(self.list_max_abs_forces, 6)

    parprint('+--------+------+---------------------+'
             '-------------------------+')
    parprint('| Method | Step |        Time         |   Energy    |'
             'fmax   |')
    parprint('+--------+------+---------------------'
             '+-------------------------+')
    for i in range(0, len(energy_tab)):
        parprint('| ML-Min |'
                 + '{0:6d}|'.format(i),
                 print_time()
                 + ' |{0:12f} |'.format(energy_tab[i])
                 + '{0:10f} |'.format(fmax_tab[i]))
    parprint('+--------+------+---------------------'
             '+-------------------------+')


def store_results_neb(self):
    """ Function that dumps the predicted discrete and interpolated M.E.P.
        curves in csv files for plotting"""

    # Save discrete path:
    header_data = 'Path distance (Angstrom), Energy (eV), Uncertainty (eV)'
    data = np.zeros((len(self.s), 3))
    data[:, 0] = self.s
    data[:, 1] = self.e
    data[:, 2] = self.uncertainty_path
    np.savetxt('results_neb.csv', X=data, header=header_data)

    # Save interpolated path:
    header_data = 'Path distance (Angstrom), Energy (eV)'
    data = np.zeros((len(self.sfit), 2))
    data[:, 0] = self.sfit
    data[:, 1] = self.efit
    np.savetxt('results_neb_interpolation.csv', X=data, header=header_data)


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
    parprint(""" 
       ____      _   _                          
      / ___|__ _| |_| |    ___  __ _ _ __ _ __  
     | |   / _` | __| |   / _ \/ _` | '__| '_ \ 
     | |__| (_| | |_| |__|  __/ (_| | |  | | | |
      \____\__,_|\__|_____\___|\__,_|_|  |_| |_| """ + version + """
      
      """)


def print_cite_mlmin():
    msg = "-----------------------------------------------------------"
    msg += "-----------------------------------------------------------\n"
    msg += "You are using ML-Min and CatLearn. Please cite: \n"
    msg += "[1] M. H. Hansen, J. A. Garrido Torres, P. C. Jennings, "
    msg += "Z. Wang, J. R. Boes, O. G. Mamun and T. Bligaard. "
    msg += "An Atomistic Machine Learning Package"
    msg += "for Surface Science and Catalysis. "
    msg += "https://arxiv.org/abs/1904.00904 \n"
    msg += "[2] J. A. Garrido Torres, M. H. Hansen, P. C. Jennings, "
    msg += "J. R. Boes and T. Bligaard. Phys. Rev. Lett. 122, 156001. "
    msg += "https://journals.aps.org/prl/abstract/10.1103/PhysRevLett" \
           ".122.156001 \n"
    msg += "-----------------------------------------------------------"
    msg += "-----------------------------------------------------------"
    parprint(msg)


def print_cite_mlneb():
    msg = "-----------------------------------------------------------"
    msg += "-----------------------------------------------------------\n"
    msg += "You are using ML-NEB and CatLearn. Please cite: \n"
    msg += "[1] J. A. Garrido Torres, M. H. Hansen, P. C. Jennings, "
    msg += "J. R. Boes and T. Bligaard. Phys. Rev. Lett. 122, 156001. "
    msg += "https://journals.aps.org/prl/abstract/10.1103/PhysRevLett" \
           ".122.156001 \n"
    msg += "[2] M. H. Hansen, J. A. Garrido Torres, P. C. Jennings, "
    msg += "Z. Wang, J. R. Boes, O. G. Mamun and T. Bligaard. "
    msg += "An Atomistic Machine Learning Package"
    msg += "for Surface Science and Catalysis. "
    msg += "https://arxiv.org/abs/1904.00904 \n"
    msg += "-----------------------------------------------------------"
    msg += "-----------------------------------------------------------"
    parprint(msg)
