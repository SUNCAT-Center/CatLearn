"""Script testing random order of atoms index on fingerprint vector."""
import networkx as nx
import networkx.algorithms.isomorphism as iso
import random
import os

from ase.io import read

from atoml.fingerprint.neighbor_matrix import connection_dict


def build_graph(atoms):
    """Make the NetworkX graph."""
    an = atoms.get_atomic_numbers()
    ats = list(range(len(atoms)))
    cd = connection_dict(atoms)
    G = nx.Graph()
    G.add_nodes_from(ats)
    for c in cd:
        for i in cd[c]:
            if i is not -1:
                w = float((an[c] + an[i]) / 2)
                G.add_edge(c, i, weight=w)
    return G


def atoms_shuffle(file):
    """Randomly order the atoms index."""
    with open("../data/{}".format(file), mode="r",
              encoding="utf-8") as xyzfile:
        lines = list(xyzfile)
    head, body = lines[:2], lines[2:]
    random.shuffle(body)
    lines = head + body
    with open("test_shuffle.xyz", mode="w", encoding="utf-8") as xyzfile:
        for i in lines:
            xyzfile.write("%s" % i)


# Specify two atoms objects to test against.
al = ['Au1Pt146_1.xyz', 'Au1Pt146_2.xyz']  # Au95Pt52, Au81Pt66
# Define the ordering of the tests.
dl = [[al[0], al[0]], [al[1], al[1]], [al[0], al[1]]]
# Define the correct results of the tests.
tr = [True, True, False]

for l in range(len(dl)):
    for _ in range(2):
        atoms_shuffle(dl[l][0])
        atoms = read('test_shuffle.xyz')
        graph1 = build_graph(atoms)
        os.remove('test_shuffle.xyz')

        atoms_shuffle(dl[l][1])
        atoms = read('test_shuffle.xyz')
        graph2 = build_graph(atoms)
        os.remove('test_shuffle.xyz')

        em = iso.numerical_edge_match('weight', 78.5)
        res = nx.is_isomorphic(graph1, graph2, edge_match=em)

        assert tr[l] == res

        print('is the same? {0} -- should be the same? {1}'.format(
              res, tr[l]))
