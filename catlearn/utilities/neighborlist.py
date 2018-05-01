"""Functions to generate the neighborlist."""
import numpy as np

from ase.neighborlist import NeighborList
from catlearn.fingerprint.periodic_table_data import get_radius


def ase_neighborlist(atoms, cutoffs=None):
    """Make dict of neighboring atoms using ase function.

    This provides a wrapper for the ASE neighborlist generator. Currently
    default values are used.

    Parameters
    ----------
    atoms : object
        Target ase atoms object on which to get neighbor list.
    cutoffs : list
        A list of radii for each atom in atoms.
    rtol : float
        The tolerance factor to allow for small variation in the cutoff radii.

    Returns
    -------
    neighborlist : dict
        A dictionary containing the atom index and each neighbor index.
    """
    if cutoffs is None:
        cutoffs = [get_radius(a.number) for a in atoms]
    nl = NeighborList(
        cutoffs, skin=0., sorted=False, self_interaction=False,
        bothways=True)

    nl.update(atoms)

    neighborlist = {}
    for i, _ in enumerate(atoms):
        neighborlist[i] = sorted(list(map(int, nl.get_neighbors(i)[0])))

    return neighborlist


def catlearn_neighborlist(atoms, dx=None, max_neighbor=1, mic=True):
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
    max_neighbor : int or str
        Maximum neighbor shell. If int is passed this will define how many
        shells to consider. If 'full' is passed then all neighbor combinations
        will be included. This might get expensive for particularly large
        systems.

    Returns
    -------
    connection_matrix : array
        An array of the neighbor shell each atom index is located in.
    """
    atomic_numbers = atoms.get_atomic_numbers()

    # Set up buffer dict.
    if dx is None:
        dx = dict.fromkeys(set(atomic_numbers), 0)
        for i in dx:
            dx[i] = get_radius(i) / 5.

    dist = atoms.get_all_distances(mic=mic)

    r_list, b_list = [], []
    for an in atomic_numbers:
        r_list.append(get_radius(an))
        b_list.append(dx[an])

    radius_matrix = np.asarray(r_list) + np.reshape(r_list, (len(r_list), 1))
    buffer_matrix = np.asarray(b_list) + np.reshape(
        b_list, (len(b_list), 1)) / 2.

    connection_matrix = np.zeros((len(atoms), len(atoms)))
    if isinstance(max_neighbor, int):
        for n in range(max_neighbor):
            dist, connection_matrix, unconnected = _neighbor_iterator(
                dist, radius_matrix, buffer_matrix, n, connection_matrix)
    elif max_neighbor == 'full':
        n, unconnected = 0, 1
        while unconnected > 0:
            dist, connection_matrix, unconnected = _neighbor_iterator(
                dist, radius_matrix, buffer_matrix, n, connection_matrix)
            n += 1
    else:
        msg = 'max_neighbor parameter {} not recognized.'.format(max_neighbor)
        raise NotImplementedError(msg)

    np.fill_diagonal(connection_matrix, 0.)

    return connection_matrix


def _neighbor_iterator(dist, radius_matrix, buffer_matrix, n,
                       connection_matrix):
    """An iterator function to work out neighbor shells.

    Parameters
    ----------
    dist : array
        Distance matrix for the atoms object.
    radius_matrix : array
        An NxN matrix of summed radii between each pair of atoms.
    buffer_matrix : array
        An NxN matrix of buffer values between each pair of atoms.
    n : int
        The current neighbor shell.
    connection_matrix : array
        The current connection matrix being built.

    Returns
    -------
    dist : array
        Modified distance matrix for the atoms object. Modified to avoid
        recounting neighbors.
    connection_matrix : array
        Modified current connection matrix. Contains results for current shell.
    unconnected : int
        The number of atom pairs that dont have a neighbor shell assigned.
    """
    res = dist - ((radius_matrix + buffer_matrix) * (n + 1))

    res[res > 0.] = 0.
    res[res < 0.] = n + 1.

    dist += res * 10000.

    connection_matrix += res
    unconnected = len(connection_matrix[connection_matrix == 0.])

    return dist, connection_matrix, unconnected


def ase_connectivity(atoms, cutoffs=None, count_bonds=True):
    """Return a connectivity matrix calculated of an atoms object.

    If no neighborlist or connectivity matrix is attached to the atoms object,
    a new one will be generated. Multiple connections are counted.

    Parameters
    ----------
    atoms : object
        An ase atoms object.
    cutoffs : list
        A list of cutoff radii for the atoms, ordered by atom index.

    Returns
    -------
    conn : array
        An n by n, where n is len(atoms).
    """
    if hasattr(atoms, 'connectivity'):
        return atoms.connectivity

    if hasattr(atoms, 'neighborlist'):
        nl = atoms.neighborlist
    else:
        nl = ase_neighborlist(atoms, cutoffs=cutoffs)

    conn_mat = []
    index = range(len(atoms))
    # Create binary matrix denoting connections.
    for index1 in index:
        conn_x = []
        for index2 in index:
            if index2 in nl[index1]:
                if count_bonds:
                    bonds = nl[index1].count(index2)
                else:
                    bonds = 1
                conn_x.append(bonds)
            else:
                conn_x.append(0.)
        conn_mat.append(conn_x)

    return np.asarray(conn_mat, dtype=int)
