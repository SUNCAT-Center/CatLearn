from math import sqrt
from ase import Atoms, Atom
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from ase.neb import NEB
from ase.io import read, write
from catlearn.optimize.catlearn_neb_optimizer import CatLearnNEB
import matplotlib.pyplot as plt
from ase.neb import NEBTools
import copy

""" 
    Toy model rearrangement of CO on Cu(111).
    This example contains: 
    1. Optimization of the initial and final end-points of the reaction path. 
    2.A. NEB optimization using CI-NEB as implemented in ASE. 
    2.B. NEB optimization using our machine-learning surrogate model.
    3) Comparison between the ASE NEB and our Machine Learning assisted NEB 
       algorithm.
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

images = [slab]
for i in range(n_images-1):
    image = slab.copy()
    # Set constraints and calculator:
    image.set_constraint(constraint)
    image.calc = copy.deepcopy(ase_calculator)
    images.append(image)

# Displace last image:
image[-2].position = image[-1].position
image[-1].x = d
image[-1].y = d / sqrt(3)

dyn = BFGS(images[-1])
dyn.run(fmax=0.05)
neb = NEB(images, climb=True)

# Interpolate positions between initial and final states:
neb.interpolate(method='idpp')

for image in images:
    print(image.positions[-1], image.get_potential_energy())

dyn = BFGS(neb, maxstep=0.04, trajectory='neb_ase.traj')
dyn.run(fmax=0.05)

for image in images:
    print(image.positions[-1], image.get_potential_energy())

nebtools_ase = NEBTools(images)

Sf_ase = nebtools_ase.get_fit()[2]
Ef_ase = nebtools_ase.get_fit()[3]

Ef_neb_ase, dE_neb_ase = nebtools_ase.get_barrier(fit=False)
nebtools_ase.plot_band()

plt.show()

# 2.B. NEB using CatLearn ####################################################
write('initial.traj', images[0])
write('final.traj', images[-1])

neb_catlearn = CatLearnNEB(start='initial.traj',
                           end='final.traj',
                           ase_calc=copy.deepcopy(ase_calculator),
                           n_images=n_images,
                           interpolation='idpp', restart=False)

neb_catlearn.run(fmax=0.05, plot_neb_paths=True)


# 3. Summary of the results #################################################

# NEB ASE:
print('\nSummary of the results: \n')

atoms_ase = read('neb_ase.traj', ':')
n_eval_ase = len(atoms_ase) - 2 * n_images

print('Number of function evaluations CI-NEB implemented in ASE:', n_eval_ase)

# Catlearn:
atoms_catlearn = read('evaluated_structures.traj', ':')
n_eval_catlearn = len(atoms_catlearn) - 2
print('Number of function evaluations CatLearn:', n_eval_catlearn)

# Comparison:
print('\nThe CatLearn algorithm required ',
      (n_eval_ase/n_eval_catlearn),
      'times less number of function evaluations than '
      'the standard NEB algorithm.')
