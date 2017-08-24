"""Script testing random order of atoms index on fingerprint vector."""
import numpy as np
import random
import os

from ase.io import read

from atoml.neighbor_matrix import connection_matrix


def atoms_shuffle():
    """Randomly order the atoms index."""
    with open("../data/Au95Pt52.xyz", mode="r", encoding="utf-8") as xyzfile:
        lines = list(xyzfile)
    head, body = lines[:2], lines[2:]
    random.shuffle(body)
    lines = head + body
    with open("test_shuffle.xyz", mode="w", encoding="utf-8") as xyzfile:
        for i in lines:
            xyzfile.write("%s" % i)


res_store = None
same_base = True
for i in range(5):
    atoms_shuffle()
    atoms = read('test_shuffle.xyz')
    c1 = connection_matrix(atoms=atoms, dx=0.4, neighbor_number=1)

    w, v = np.linalg.eig((np.array(c1)))
    r1 = np.sort(w)[::-1]

    if res_store is not None:
        same = np.allclose(res_store[1], c1)
        if not same:
            same_base = same
        assert np.allclose(res_store[0], r1)
    res_store = [r1, c1]
assert not same_base

os.remove('test_shuffle.xyz')
