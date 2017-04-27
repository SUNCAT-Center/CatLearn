""" Functions to build a graph based on the neighbor list. """
from __future__ import absolute_import
from __future__ import division

import numpy as np
from mendeleev import element

from ase.ga.utilities import get_neighborlist


def connection_matrix(atoms, dx=0.2):
    """ Helper function to generate a connections matrix from an atoms object.

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

    cm = []
    r = range(len(atoms))
    # Create binary matrix denoting connections.
    for i in r:
        x = []
        for j in r:
            if j in nl[i]:
                x.append(1.)
            else:
                x.append(0.)
        cm.append(x)

    return np.asarray(cm)


def element_list(an, no):
    """ Function to transform list of atom numbers into binary list with one
        for given type, zero for all others. Maps homoatomic interactions.

        Parameters
        ----------
        an : list
            List of atom numbers.
        no : int
            Select atom number.
    """
    hm = []
    for n in an:
        if n == no:
            hm.append(1.)
        else:
            hm.append(0.)

    return hm


def heteroatomic_matrix(an, el):
    """ Function to transform list of atom numbers into binary list with one
        for interactions between two different types, zero for all others. Maps
        heteroatomic interactions.

        Parameters
        ----------
        an : list
            List of atom numbers.
        el : list
            List of two atom numbers on which to map interactions.
    """
    hm = []
    for i in an:
        if i == el[0]:
            x = []
            for j in an:
                if j != el[0]:
                    x.append(1.)
                else:
                    x.append(0.)
        else:
            x = [0] * len(an)
        hm.append(x)

    return np.asarray(hm)


def generalized_matrix(cm):
    """ Get the generalized coordination matrix.

        Parameters
        ----------
        cm : array
            The connections matrix.
    """
    gm = []
    for i in cm:
        tot = 0.
        for j in range(len(i)):
            if i[j] == 1.:
                tot += sum(i)
        gm.append(tot / 12.)

    return np.asarray(gm)


def property_matrix(atoms, property):
    """ Generate a property matrix based on the atomic types.

        Parameters
        ----------
        atoms : object
            The target ase atoms opject.
        property : str
            The target property from mendeleev.
    """
    sy = atoms.get_chemical_symbols()
    ce = {}
    for s in set(sy):
        ce[s] = eval('element("' + s + '").' + property)

    x = []
    for s in sy:
        x.append(ce[s])
    pm = [x] * len(atoms)

    return np.asarray(np.float64(pm))


def base_f(atoms, property=None):
    """ Function to generate features from atoms objects.

        Parameters
        ----------
        atoms : object
            The target ase atoms object.
    """
    fp = []

    # Generate the required data from atoms object.
    cm = connection_matrix(atoms, dx=0.2)
    if property is not None:
        pm = property_matrix(atoms=atoms, property=property)
        cm *= pm
    scm = np.sum(cm, axis=1)
    gm = generalized_matrix(cm)
    an = atoms.get_atomic_numbers()

    # Get level one fingerprint. Sum of coordination for each atom type.
    done = []
    for e in set(an):
        el = element_list(an, e)
        fp.append(np.sum(np.array(scm) * np.array(el)))
        fp.append(np.sum((np.array(scm) * np.array(el)) ** 2))
        fp.append(np.sum((np.array(scm) * np.array(el)) ** 0.5))

        # Get level two fingerprint. Total AA, AB, BB etc bonds.
        pt = np.array(([el] * len(atoms)))
        em = np.sum(np.sum(pt * np.array(cm), axis=1))
        fp.append(em)
        if e not in done:
            done.append(e)
        for eo in set(an):
            if eo not in done:
                hm = heteroatomic_matrix(an, [e, eo])
                fp.append(np.sum(np.sum(np.array(hm) * np.array(cm), axis=1)))

        # Get level three fingerprint. Generalized coordination number.
        fp.append(np.sum(np.array(gm) * np.array(el)))
        fp.append(np.sum((np.array(gm) * np.array(el)) ** 2))
        fp.append(np.sum((np.array(gm) * np.array(el)) ** 0.5))

    return np.asarray(fp)
