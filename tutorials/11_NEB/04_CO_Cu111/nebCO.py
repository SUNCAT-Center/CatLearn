from math import sqrt
from ase import Atoms, Atom
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from ase.neb import NEB
from ase.io import read, write
from catlearn.optimize.mlneb import MLNEB
import matplotlib.pyplot as plt
from ase.neb import NEBTools
import copy
from catlearn.optimize.tools import plotneb


""" 
    Toy model rearrangement of CO on Cu(111).
    This example contains: 
    1. Optimization of the initial and final end-points of the reaction path. 
    2.A. NEB optimization using CI-NEB as implemented in ASE. 
    2.B. NEB optimization using our machine-learning surrogate model.
    3. Comparison between the ASE NEB and our ML-NEB algorithm.
"""

# Calculator:
ase_calculator = EMT()
n_images = 11

# 1. Structural relaxation. ##################################################

# Distance between Cu atoms on a (111) surface:
a = 3.6
d = a / sqrt(2)
fcc111 = Atoms(symbols='Cu',
               cell=[(d, 0, 0),
                     (d / 2, d * sqrt(3) / 2, 0),
                     (d / 2, d * sqrt(3) / 6, -a / sqrt(3))],
               pbc=True)
slab = fcc111 * (2, 2, 4)
slab.set_cell([2 * d, d * sqrt(3), 1])
slab.set_pbc((1, 1, 0))
slab.calc = EMT()
Z = slab.get_positions()[:, 2]
indices = [i for i, z in enumerate(Z) if z < Z.mean()]
constraint = FixAtoms(indices=indices)
slab.set_constraint(constraint)
dyn = BFGS(slab)
dyn.run(fmax=0.01)
Z = slab.get_positions()[:, 2]


b = 1.2
h = 1.5
slab += Atom('C', (d / 2, -b / 2, h))
slab += Atom('O', (d / 2, +b / 2, h))
s = slab.copy()
dyn = BFGS(slab)
dyn.run(fmax=0.01)


# 2.A. NEB using ASE #########################################################

images_ase = [slab]
for i in range(n_images-1):
    image = slab.copy()
    # Set constraints and calculator:
    image.set_constraint(constraint)
    image.calc = copy.deepcopy(ase_calculator)
    images_ase.append(image)

# Displace last image:
image[-2].position = image[-1].position
image[-1].x = d
image[-1].y = d / sqrt(3)

dyn = BFGS(images_ase[-1])
dyn.run(fmax=0.05)
neb = NEB(images_ase, climb=True)

# Interpolate positions between initial and final states:
neb.interpolate(method='idpp')

for image in images_ase:
    print(image.positions[-1], image.get_potential_energy())

dyn = BFGS(neb, maxstep=0.04, trajectory='neb_ase.traj')
dyn.run(fmax=0.05)

for image in images_ase:
    print(image.positions[-1], image.get_potential_energy())

# 2.B. NEB using CatLearn

write('initial.traj', images_ase[0])
write('final.traj', images_ase[-1])

neb_catlearn = MLNEB(start='initial.traj',
                     end='final.traj',
                     ase_calc=copy.deepcopy(ase_calculator),
                     n_images=n_images,
                     interpolation='idpp', restart=False)

neb_catlearn.run(fmax=0.05, trajectory='ML-NEB.traj')

# 3. Summary of the results

# NEB ASE:
print('\nSummary of the results: \n')

atoms_ase = read('neb_ase.traj', ':')
n_eval_ase = len(atoms_ase) - 2 * (len(atoms_ase)/n_images)

print('Number of function evaluations CI-NEB implemented in ASE:', n_eval_ase)

# ML-NEB:
atoms_catlearn = read('evaluated_structures.traj', ':')
n_eval_catlearn = len(atoms_catlearn) - 2
print('Number of function evaluations CatLearn:', n_eval_catlearn)

# Comparison:
print('\nThe ML-NEB algorithm required ',
      (n_eval_ase/n_eval_catlearn),
      'times less number of function evaluations than '
      'the standard NEB algorithm.')

# Plot ASE NEB:
nebtools_ase = NEBTools(images_ase)

Sf_ase = nebtools_ase.get_fit()[2]
Ef_ase = nebtools_ase.get_fit()[3]

Ef_neb_ase, dE_neb_ase = nebtools_ase.get_barrier(fit=False)
nebtools_ase.plot_band()

plt.show()

# Plot ML-NEB predicted path and show images along the path:
plotneb(trajectory='ML-NEB.traj', view_path=False)