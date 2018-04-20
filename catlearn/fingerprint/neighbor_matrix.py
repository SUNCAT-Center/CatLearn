"""Functions to build a neighbor matrix feature representation."""
from __future__ import absolute_import
from __future__ import division

import json
import numpy as np
from collections import defaultdict

from ase.data import covalent_radii
from ase.ga.utilities import get_mic_distance

from catlearn import __path__ as catlearn_path


def neighbor_features(atoms, property=None, periodic=False, dx=0.2,
                      neighbor_number=1, reuse_nl=False):
    """Generate predefined features from atoms objects.

    Parameters
    ----------
    atoms : object
        The target ase atoms object.
    property : list
        List of the target properties from mendeleev.
    periodic : boolean
        Specify whether to use the periodic neighborlist generator. None
        periodic method is faster and used by default.
    dx : float
        Buffer to calculate nearest neighbor pairs.
    neighbor_number : int
        Neighbor shell.
    reuse_nl : boolean
        Whether to reuse a previously stored neighborlist if available.
    """
    features = []

    # Generate the required data from atoms object.
    an = atoms.get_atomic_numbers()
    conn_mat_store = connection_matrix(atoms=atoms, periodic=periodic, dx=dx,
                                       neighbor_number=neighbor_number,
                                       reuse_nl=reuse_nl)
    sum_conn_mat = np.sum(conn_mat_store, axis=1)
    gen_mat = _generalized_matrix(conn_mat_store)

    features += _get_features(an=an, conn_mat=conn_mat_store,
                              sum_cm=sum_conn_mat, gen_mat=gen_mat)

    if property is not None:
        for p in property:
            prop_mat = property_matrix(atoms=atoms, property=p)
            conn_mat = conn_mat_store * prop_mat
            sum_cm = np.sum(conn_mat, axis=1)
            gen_cm = _generalized_matrix(conn_mat)

            features += _get_features(an=an, conn_mat=conn_mat, sum_cm=sum_cm,
                                      gen_mat=gen_cm)

    return np.asarray(features)


def connection_matrix(atoms, periodic=False, dx=0.2, neighbor_number=1,
                      reuse_nl=False):
    """Generate a connections matrix from an atoms object.

    Parameters
    ----------
    atoms : object
        Target ase atoms object on which to build the connections matrix.
    periodic : boolean
        Specify whether to use the periodic neighborlist generator. None
        periodic method is faster and used by default.
    dx : float
        Buffer to calculate nearest neighbor pairs.
    neighbor_number : int
        Neighbor shell.
    reuse_nl : boolean
        Whether to reuse a previously stored neighborlist if available.
    """
    # Use ase.ga neighbor list generator.
    if reuse_nl and 'neighborlist' in atoms.info['key_value_pairs']:
        nl = atoms.info['key_value_pairs']['neighborlist']
    elif periodic:
        nl = _get_periodic_neighborlist(atoms, dx=dx,
                                        neighbor_number=neighbor_number)
    else:
        nl = _get_neighborlist(atoms, dx=dx, neighbor_number=neighbor_number)

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


def connection_dict(atoms, periodic=False, dx=0.2, neighbor_number=1,
                    reuse_nl=False):
    """Generate a dict of atom connections.

    Parameters
    ----------
    atoms : object
        Target ase atoms object on which to build the connections matrix.
    periodic : boolean
        Specify whether to use the periodic neighborlist generator. None
        periodic method is faster and used by default.
    dx : float
        Buffer to calculate nearest neighbor pairs.
    neighbor_number : int
        Neighbor shell.
    reuse_nl : boolean
        Whether to reuse a previously stored neighborlist if available.
    """
    # Use ase.ga neighbor list generator.
    if reuse_nl and 'neighborlist' in atoms.info['key_value_pairs']:
        nl = atoms.info['key_value_pairs']['neighborlist']
    elif periodic:
        nl = _get_periodic_neighborlist(atoms, dx=dx)
    else:
        nl = _get_neighborlist(atoms, dx=dx)

    # Pad neighborlist with negative one.
    mlen = max([len(n) for n in nl.values()])
    for i in nl:
        if len(nl[i]) < mlen:
            nl[i] += [-1] * (mlen - len(nl[i]))

    return nl


def property_matrix(atoms, property):
    """Generate a property matrix based on the atomic types.

    Parameters
    ----------
    atoms : object
        The target ase atoms opject.
    property : str
        The target property from mendeleev.
    """
    # Load the Mendeleev parameter data into memory
    with open('/'.join(catlearn_path[0].split('/')[:-1]) +
              '/catlearn/data/proxy-mendeleev.json') as f:
        data = json.load(f)

    an = atoms.get_atomic_numbers()
    atomic_prop = {}
    for a in set(an):
        atomic_prop[a] = data[str(a)].get(property)

    prop_x = []
    for a in an:
        prop_x.append(atomic_prop[a])
    prop_mat = [prop_x] * len(atoms)

    return np.asarray(np.float64(prop_mat))


def _get_neighborlist(atoms, dx=0.2, neighbor_number=1):
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
                d = np.linalg.norm(np.asarray(a1.position) -
                                   np.asarray(a2.position))
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


def _get_periodic_neighborlist(atoms, dx=0.2, neighbor_number=1):
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


def _element_list(an, no):
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


def _heteroatomic_matrix(an, el):
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


def _generalized_matrix(conn_mat):
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


def _get_features(an, conn_mat, sum_cm, gen_mat):
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
        el = _element_list(an, e)
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
                hm = _heteroatomic_matrix(an, [e, eo])
                feature.append(np.sum(np.sum(np.array(hm) * np.array(conn_mat),
                                             axis=1)))

        # Get level three fingerprint. Generalized coordination number.
        x = np.array(gen_mat) * np.array(el)
        feature.append(np.sum(x))
        feature.append(np.sum(x ** 2))
        feature.append(np.sum(x ** 0.5))

    return feature
