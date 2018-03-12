"""Functions to generate the neighborlist."""
import numpy as np

from ase.neighborlist import NeighborList
from ase.data import covalent_radii
from atoml.fingerprint.periodic_table_data import get_radius


def ase_neighborlist(atoms):
    """Make dict of neighboring atoms using ase function."""
    cutoffs = [covalent_radii[a.number] for a in atoms]
    nl = NeighborList(
        cutoffs, skin=0.3, sorted=False, self_interaction=False, bothways=True)

    nl.build(atoms)

    neighborlist = {}
    for i, _ in enumerate(atoms):
        neighborlist[i] = sorted(list(map(int, nl.get_neighbors(i)[0])))

    return neighborlist


def atoml_neighborlist(atoms, dx=None, max_neighbor=1):
    """Make dict of neighboring atoms for discrete system.

    Possible to return neighbors from defined neighbor shell e.g. 1st, 2nd,
    3rd by changing the neighbor number.

    Parameters
    ----------
    atoms : object
        Target ase atoms object on which to get neighbor list.
    dx : dict
        Buffer to calculate nearest neighbor pairs in dict format:
        dx = {atomic_number: buffer}.
    max_neighbor : int
        Neighbor shell.
    """
    atomic_numbers = atoms.get_atomic_numbers()

    # Set up buffer dict.
    if dx is None:
        dx = dict.fromkeys(set(atomic_numbers), 0)
        for i in dx:
            dx[i] = get_radius(i) / 10.

    dist = atoms.get_all_distances(mic=True)

    r_list, b_list = [], []
    for an in atomic_numbers:
        r_list.append(get_radius(an))
        b_list.append(dx[an])

    radius_matrix = np.asarray(r_list) + np.reshape(r_list, (len(r_list), 1))
    buffer_matrix = np.asarray(b_list) + np.reshape(b_list, (len(b_list), 1))

    connection_matrix = np.zeros((len(atoms), len(atoms)))
    if type(max_neighbor) == int:
        for n in range(max_neighbor):
            connection_matrix += _neighbor_iterator(
                dist, radius_matrix, buffer_matrix, n, connection_matrix)[0]

    np.fill_diagonal(connection_matrix, np.asarray(atomic_numbers, dtype='f'))

    return connection_matrix


def _neighbor_iterator(dist, radius_matrix, buffer_matrix, n,
                       connection_matrix):
    res = dist - ((radius_matrix + buffer_matrix) * (n + 1))

    res[res > 0.] = 0.
    res[res < 0.] = n + 1.

    dist += res * 10000.

    connection_matrix += res
    unconnected = len(connection_matrix[connection_matrix == 0.])

    return connection_matrix, unconnected
