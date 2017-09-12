"""Script to visualize graph features."""
import random
import os

from ase.io import read

from atoml.neighbor_matrix import connection_matrix, property_matrix

import matplotlib.pyplot as plt
from matplotlib import cm as CM


def atoms_shuffle():
    """Randomly order the atoms index."""
    with open("../../data/Au95Pt52.xyz", mode="r") as xyzfile:
        lines = list(xyzfile)
    head, body = lines[:2], lines[2:]
    random.shuffle(body)
    lines = head + body
    with open("test_shuffle.xyz", mode="w", encoding="utf-8") as xyzfile:
        for i in lines:
            xyzfile.write("%s" % i)


atoms_shuffle()
atoms = read('test_shuffle.xyz')

pm = property_matrix(atoms=atoms, property='melting_point')
pm *= property_matrix(atoms=atoms, property='atomic_volume')

c1 = connection_matrix(atoms=atoms, dx=0.4, neighbor_number=1)
c2 = connection_matrix(atoms=atoms, dx=0.4, neighbor_number=2)
c3 = connection_matrix(atoms=atoms, dx=0.4, neighbor_number=3)
c4 = connection_matrix(atoms=atoms, dx=0.4, neighbor_number=4)
c5 = connection_matrix(atoms=atoms, dx=0.4, neighbor_number=5)
c6 = connection_matrix(atoms=atoms, dx=0.4, neighbor_number=6)

all = (c1*6) + (c2*5) + (c3*4) + (c4*3) + (c5*2) + (c6*1)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(231)
ax.imshow(c1, cmap=CM.binary)
ax = fig.add_subplot(232)
ax.imshow(c2, cmap=CM.binary)
ax = fig.add_subplot(233)
ax.imshow(c3, cmap=CM.binary)
ax = fig.add_subplot(234)
ax.imshow(c4, cmap=CM.binary)
ax = fig.add_subplot(235)
ax.imshow(c5, cmap=CM.binary)
ax = fig.add_subplot(236)
ax.imshow(c6, cmap=CM.binary)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(121)
ax.imshow(all, cmap=CM.YlGnBu)
ax = fig.add_subplot(122)
ax.imshow(all*pm, cmap=CM.YlGnBu)

plt.show()

os.remove('test_shuffle.xyz')
