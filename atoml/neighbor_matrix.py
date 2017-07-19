"""Functions to build a neighbor matrix feature representation."""
from __future__ import absolute_import
from __future__ import division

import numpy as np
from collections import defaultdict
from mendeleev import element

from ase.data import covalent_radii
from ase.ga.utilities import get_mic_distance


def get_neighborlist(atoms, dx=0.2, neighbor_number=1):
    """Make dict of neighboring atoms for discrete system.

    Possible to return neighbors from defined neighbor shell e.g. 1st, 2nd,
    3rd by changing the neighbor number.

    Parameters
    ----------
    atoms : object
        Target ase atoms object on which to get neighbor list.
    dx : float
        Buffer to calculate nearest neighbor pairs.
    neighbor_number : int
        Neighbor shell.
    """
    conn = defaultdict(list)
    for a1 in atoms:
        for a2 in atoms:
            if a1.index != a2.index:
                d = np.linalg.norm(np.asarray(a1.position)
                                   - np.asarray(a2.position))
                r1 = covalent_radii[a1.number]
                r2 = covalent_radii[a2.number]
                if neighbor_number == 1:
                    d_max1 = 0.
                else:
                    d_max1 = ((neighbor_number - 1) * (r2 + r1)) + dx
                d_max2 = (neighbor_number * (r2 + r1)) + dx
                if d > d_max1 and d < d_max2:
                    conn[a1.index].append(a2.index)
    return conn


def get_periodic_neighborlist(atoms, dx=0.2, neighbor_number=1):
    """Make dict of neighboring atoms for periodic system.

    Possible to return neighbors from defined neighbor shell e.g. 1st, 2nd,
    3rd by changing the neighbor number.

    Parameters
    ----------
    atoms : object
        Target ase atoms object on which to get neighbor list.
    dx : float
        Buffer to calculate nearest neighbor pairs.
    neighbor_number : int
        Neighbor shell.
    """
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()

    conn = defaultdict(list)
    for atomi in atoms:
        for atomj in atoms:
            if atomi.index != atomj.index:
                d = get_mic_distance(atomi.position,
                                     atomj.position,
                                     cell,
                                     pbc)
                cri = covalent_radii[atomi.number]
                crj = covalent_radii[atomj.number]
                if neighbor_number == 1:
                    d_max1 = 0.
                else:
                    d_max1 = ((neighbor_number - 1) * (crj + cri)) + dx
                d_max2 = (neighbor_number * (crj + cri)) + dx
                if d > d_max1 and d < d_max2:
                    conn[atomi.index].append(atomj.index)
    return conn


def connection_matrix(atoms, dx=0.2):
    """Helper function to generate a connections matrix from an atoms object.

    Parameters
    ----------
    atoms : object
        Target ase atoms object on which to build the connections matrix.
    dx : float
        Buffer to calculate nearest neighbor pairs.
    """
    # Use ase.ga neighbor list generator.
    if 'neighborlist' in atoms.info['key_value_pairs']:
        nl = atoms.info['key_value_pairs']['neighborlist']
    else:
        nl = get_neighborlist(atoms, dx=dx)

    conn_mat = []
    index = range(len(atoms))
    # Create binary matrix denoting connections.
    for index1 in index:
        conn_x = []
        for index2 in index:
            if index2 in nl[index1]:
                conn_x.append(1.)
            else:
                conn_x.append(0.)
        conn_mat.append(conn_x)

    return np.asarray(conn_mat)


def element_list(an, no):
    """Binary mapping of homoatomic interactions.

    Parameters
    ----------
    an : list
        List of atom numbers.
    no : int
        Select atom number.
    """
    binary_inter = []
    for n in an:
        if n == no:
            binary_inter.append(1.)
        else:
            binary_inter.append(0.)

    return binary_inter


def heteroatomic_matrix(an, el):
    """Binary mapping of heteroatomic interactions.

    Parameters
    ----------
    an : list
        List of atom numbers.
    el : list
        List of two atom numbers on which to map interactions.
    """
    binary_inter = []
    for i in an:
        if i == el[0]:
            inter_x = []
            for j in an:
                if j != el[0]:
                    inter_x.append(1.)
                else:
                    inter_x.append(0.)
        else:
            inter_x = [0] * len(an)
        binary_inter.append(inter_x)

    return np.asarray(binary_inter)


def generalized_matrix(conn_mat):
    """Get the generalized coordination matrix.

    Parameters
    ----------
    cm : array
        The connections matrix.
    """
    gen_mat = []
    for i in conn_mat:
        tot = 0.
        for j in range(len(i)):
            if i[j] != 0.:
                tot += sum(i)
        gen_mat.append(tot / 12.)

    return np.asarray(gen_mat)


def property_matrix(atoms, property):
    """Generate a property matrix based on the atomic types.

    Parameters
    ----------
    atoms : object
        The target ase atoms opject.
    property : str
        The target property from mendeleev.
    """
    symb = atoms.get_chemical_symbols()
    atomic_prop = {}
    for s in set(symb):
        atomic_prop[s] = eval('element("' + s + '").' + property)

    prop_x = []
    for s in symb:
        prop_x.append(atomic_prop[s])
    prop_mat = [prop_x] * len(atoms)

    return np.asarray(np.float64(prop_mat))


def get_features(an, conn_mat, sum_cm, gen_mat):
    """Function to generate the actual feature vector.

    Parameters
    ----------
    an : list
        Ordered list of atomic numbers.
    cm : array
        The coordination matrix.
    sum_cm : list
        The summed vectors of the coordination matrix.
    gen_cm : array
        The generalized coordination matrix.
    """
    feature = []
    # Get level one fingerprint. Sum of coordination for each atom type.
    done = []
    for e in set(an):
        el = element_list(an, e)
        x = np.array(sum_cm) * np.array(el)
        feature.append(np.sum(x))
        feature.append(np.sum(x ** 2))
        feature.append(np.sum(x ** 0.5))

        # Get level two fingerprint. Total AA, AB, BB etc bonds.
        pt = np.array(([el] * len(an)))
        em = np.sum(np.sum(pt * np.array(conn_mat), axis=1))
        feature.append(em)
        if e not in done:
            done.append(e)
        for eo in set(an):
            if eo not in done:
                hm = heteroatomic_matrix(an, [e, eo])
                feature.append(np.sum(np.sum(np.array(hm) * np.array(conn_mat),
                                             axis=1)))

        # Get level three fingerprint. Generalized coordination number.
        x = np.array(gen_mat) * np.array(el)
        feature.append(np.sum(x))
        feature.append(np.sum(x ** 2))
        feature.append(np.sum(x ** 0.5))

    return feature


def base_f(atoms, property=None):
    """Function to generate features from atoms objects.

    Parameters
    ----------
    atoms : object
        The target ase atoms object.
    property : list
        List of the target properties from mendeleev.
    """
    features = []

    # Generate the required data from atoms object.
    an = atoms.get_atomic_numbers()
    conn_mat_store = connection_matrix(atoms, dx=0.2)
    sum_conn_mat = np.sum(conn_mat_store, axis=1)
    gen_mat = generalized_matrix(conn_mat_store)

    features += get_features(an=an, conn_mat=conn_mat_store,
                             sum_cm=sum_conn_mat, gen_mat=gen_mat)

    if property is not None:
        for p in property:
            prop_mat = property_matrix(atoms=atoms, property=p)
            conn_mat = conn_mat_store * prop_mat
            sum_cm = np.sum(conn_mat, axis=1)
            gen_cm = generalized_matrix(conn_mat)

            features += get_features(an=an, conn_mat=conn_mat, sum_cm=sum_cm,
                                     gen_mat=gen_cm)

    return np.asarray(features)
