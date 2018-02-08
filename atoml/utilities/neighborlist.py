"""Functions to generate the neighborlist."""
import numpy as np
from collections import defaultdict

from ase.neighborlist import NeighborList
from ase.data import covalent_radii


def ase_neighborlist(atoms):
    """Make dict of neighboring atoms using ase function."""
    cutoffs = [covalent_radii[a.number] for a in atoms]
    nl = NeighborList(
        cutoffs, skin=0.3, sorted=False, self_interaction=False, bothways=True)

    nl.build(atoms)

    neighborlist = {}
    for i, _ in enumerate(atoms):
        neighborlist[i] = nl.get_neighbors(i)[0]

    return neighborlist


def atoms_neighborlist(atoms, dx=None, neighbor_number=1):
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
    neighbor_number : int
        Neighbor shell.
    """
    # Set up buffer dict.
    if dx is None:
        dx = dict.fromkeys(set(atoms.get_atomic_numbers()), 0)
        for i in dx:
            dx[i] = covalent_radii[i] / 2.

    conn = defaultdict(list)
    for a1 in atoms:
        for a2 in atoms:
            if a1.index != a2.index:
                d = np.linalg.norm(np.asarray(a1.position) -
                                   np.asarray(a2.position))
                r1 = covalent_radii[a1.number]
                r2 = covalent_radii[a2.number]
                dxi = (dx[a1.number] + dx[a2.number]) / 2.
                if neighbor_number == 1:
                    d_max1 = 0.
                else:
                    d_max1 = ((neighbor_number - 1) * (r2 + r1)) + dxi
                d_max2 = (neighbor_number * (r2 + r1)) + dxi
                if d > d_max1 and d < d_max2:
                    conn[a1.index].append(a2.index)
    return conn
