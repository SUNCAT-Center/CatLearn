"""Script testing random order of atoms index on fingerprint vector."""
import numpy as np
import networkx as nx
import networkx.algorithms.isomorphism as iso
import random
import os
import matplotlib.pyplot as plt

from ase.io import read

from atoml.fingerprint.neighbor_matrix import (connection_matrix,
                                               property_matrix,
                                               connection_dict)

# atoms = read('../data/Au95Pt52.xyz')
# c1 = connection_matrix(atoms=atoms, dx=0.4, neighbor_number=1)
# p1 = property_matrix(atoms=atoms, property='atomic_number')
# c1 = c1 * p1
# an = atoms.get_atomic_numbers()
# I = an * np.identity(np.shape(c1)[0])
# c1 = c1 + I


def build_graph(atoms):
    an = atoms.get_atomic_numbers()
    ats = list(range(len(atoms)))
    cd = connection_dict(atoms)
    G = nx.MultiGraph()
    G.add_nodes_from(ats)
    for c in cd:
        for i in cd[c]:
            if i is not -1:
                w = float((an[c] + an[i]) / 2)
                G.add_edge(c, i, weight=w)
    return G

# for i in range(147):
#    print(G[i])

# nx.draw(G)
# plt.show()
# exit()

# print(G.nodes())
# print(G.edges())
# exit()

atoms = read('../data/Au95Pt52.xyz')
graph1 = build_graph(atoms)
# graph1 = nx.from_numpy_matrix(c1, nx.Graph)
# graph1.edges(data=True)

# for g, a in zip(range(len(graph1)), an):
#    graph1[g]['atomic_number'] = a
#    print(graph1.neighbors(g))
#    exit()
# print(graph1.neighbors(0))
# exit()

# nx.draw(graph1)
# plt.show()
# exit()

atoms = read('../data/Au81Pt66.xyz')
graph2 = build_graph(atoms)

GM = nx.isomorphism.GraphMatcher(graph1, graph2)
print(GM.is_isomorphic())
exit()

nm = iso.numerical_edge_match(['weight', 'weight', 'weight'],
                              [78.0, 78.5, 79.0])
print(nx.is_isomorphic(graph1, graph2, nm))
exit()
c2 = connection_matrix(atoms=atoms, dx=0.4, neighbor_number=1)
p2 = property_matrix(atoms=atoms, property='atomic_number')
# c2 = c2 * p2
an = atoms.get_atomic_numbers()
I = an * np.identity(np.shape(c1)[0])
c1 = c1 + I
graph2 = nx.from_numpy_matrix(c2, nx.Graph)
graph2.edges(data=True)

assert not np.allclose(c1, c2)

print(nx.is_isomorphic(graph1, graph2), np.allclose(c1, c2))


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


al = ['Au95Pt52.xyz', 'Au81Pt66.xyz']
dl = [[al[0], al[0]], [al[1], al[1]], [al[0], al[1]]]
for l in dl:
    for _ in range(3):
        atoms_shuffle(l[0])
        atoms = read('test_shuffle.xyz')
        c1 = connection_matrix(atoms=atoms, dx=0.4, neighbor_number=1)
        p1 = property_matrix(atoms=atoms, property='atomic_number')
        # c1 = c1 * p1
        an1 = atoms.get_atomic_numbers()
        I = an1 * np.identity(np.shape(c1)[0])
        c1 = c1 + I
        graph1 = nx.from_numpy_matrix(c1)
        os.remove('test_shuffle.xyz')

        atoms_shuffle(l[1])
        atoms = read('test_shuffle.xyz')
        c2 = connection_matrix(atoms=atoms, dx=0.4, neighbor_number=1)
        p2 = property_matrix(atoms=atoms, property='atomic_number')
        # c2 = c2 * p2
        an2 = atoms.get_atomic_numbers()
        I = an2 * np.identity(np.shape(c1)[0])
        c2 = c2 + I
        graph2 = nx.from_numpy_matrix(c2)
        os.remove('test_shuffle.xyz')

        assert not np.allclose(c1, c2)

        print(nx.is_isomorphic(graph1, graph2, node_match=iso.numerical_node_match('weight', 1.0)))

        em = iso.numerical_node_match('weight', 78)
        print('is the same?', nx.is_isomorphic(graph1, graph2, em), 'should be the same?', np.allclose(np.sum(an1), np.sum(an2)))
exit()

res_store = None
same_base = True
for i in range(5):
    atoms_shuffle()
    atoms = read('test_shuffle.xyz')
    c1 = connection_matrix(atoms=atoms, dx=0.4, neighbor_number=1)

    graph1 = nx.from_numpy_matrix(c1)
    # nx.draw(graph)
    # plt.show()

    w, v = np.linalg.eig((np.array(c1)))
    r1 = np.sort(w)[::-1]

    if res_store is not None:
        same = np.allclose(res_store[1], c1)
        if not same:
            same_base = same
        msg = "Results should be the same but aren't."
        assert np.allclose(res_store[0], r1), msg
    res_store = [r1, c1]
msg = "Something went wrong and all atoms objects were identical."
assert not same_base

os.remove('test_shuffle.xyz')
