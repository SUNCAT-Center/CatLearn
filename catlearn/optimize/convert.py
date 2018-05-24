import numpy as np
from ase.io import Trajectory


def ase_traj_to_catlearn(traj_file, ase_calc=None):
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
            targets and gradients.

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
        # image_i.set_calculator(ase_calc)
        list_train = np.append(list_train, image_i.get_positions().flatten(), axis=0)
        list_targets = np.append(list_targets, image_i.get_potential_energy())
        list_gradients = np.append(list_gradients, -(image_i.get_forces(
        ).flatten()), axis=0)
    list_train = np.reshape(list_train, (number_images, num_atoms*3))
    list_targets = np.reshape(list_targets, (number_images,1))
    list_gradients = np.reshape(list_gradients, (number_images, num_atoms*3))
    results = {'first_image': images[0],
               'last_image' : images[1],
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
    atoms_pos = np.reshape(input_array,(num_atoms,3))
    x_pos = atoms_pos[:, 0]
    y_pos = atoms_pos[:, 1]
    z_pos = atoms_pos[:, 2]
    pos_ase = list(zip(x_pos, y_pos, z_pos))
    return pos_ase


def array_to_atoms(input_array):
    atoms = input_array.reshape(int(len(input_array)/3), 3)
    return atoms
