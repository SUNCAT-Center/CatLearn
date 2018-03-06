"""API to convert from ASE and NetworkX."""
from __future__ import absolute_import
from __future__ import division

import networkx as nx
from ase import Atoms

from atoml.api.ase_atoms_api import extend_atoms_class
from atoml.utilities.neighborlist import ase_neighborlist


def ase_to_networkx(atoms):
    """Make the NetworkX graph form ASE atoms object.

    The graph is dependent on the generation of the neighborlist. Currently
    this is handled by the version implemented in ASE.

    Parameters
    ----------
    atoms : object
        An ASE atoms object.

    Returns
    -------
    atoms_graph : object
        A networkx graph object.
    """
    msg = 'Please pass an ASE atoms object, not a {}'.format(type(atoms))
    assert isinstance(atoms, Atoms), msg

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


def networkx_to_adjacency(graph):
    """Simple wrapper for graph to adjacency matrix.

    Parameters
    ----------
    graph : object
        The networkx graph object.

    Returns
    -------
    matrix : array
        The numpy adjacency matrix.
    """
    msg = 'Please pass an networkx graph object, not a {}'.format(type(graph))
    assert isinstance(graph, nx.Graph), msg

    return nx.to_numpy_matrix(graph, dtype='f')
