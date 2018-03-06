"""API to convert from ASE and NetworkX."""
from __future__ import absolute_import
from __future__ import division

import networkx as nx

from atoml.api.ase_atoms_api import extend_atoms_class
from atoml.utilities.neighborlist import ase_neighborlist


def ase_to_networkx(atoms):
    """Make the NetworkX graph."""
    an = atoms.get_atomic_numbers()
    ats = list(range(len(an)))

    atoms_graph = nx.Graph()
    atoms_graph.add_nodes_from(ats)
    for n in atoms_graph:
        atoms_graph.add_node(n, atomic_number=an[n])

    extend_atoms_class(atoms)
    nl = atoms.get_neighborlist()
    if nl is None:
        nl = ase_neighborlist(atoms)

    for i in nl:
        tup = ((i, nl[i][j]) for j in range(len(nl[i])))
        atoms_graph.add_edges_from(tup)

    return atoms_graph
