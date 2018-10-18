from ase.calculators.emt import EMT
from ase.io import read
from ase.neb import NEB
from ase.optimize import BFGS, MDMin
from ase.visualize import view
import matplotlib.pyplot as plt
from catlearn.optimize.catlearn_minimizer import CatLearnMin
from catlearn.optimize.catlearn_neb_optimizer import CatLearnNEB
from ase.neb import NEBTools
import copy

""" 
    Toy model rearrangement of Pt heptamer island on Pt(111).
    This example contains: 
    1) Optimization of the initial and final end-points of the reaction path. 
    2A) NEB optimization using NEB-CI as implemented in ASE. 
    2B) NEB optimization using our machine-learning surrogate model.
    3) Comparison between the ASE NEB and our Machine Learning assisted NEB 
       algorithm.
"""

# 1. Structural relaxation. ##################################################

# Setup calculator:
ase_calculator = EMT()

# 1.1. Structures:

slab_initial = read('./A_structure/POSCAR')
slab_initial.set_calculator(copy.deepcopy(ase_calculator))

slab_final = read('./I_structure/POSCAR')
slab_final.set_calculator(ase_calculator)


# 1.2. Optimize initial and final end-points.

# Initial end-point:
qn = CatLearnMin(slab_initial, trajectory='initial.traj')
qn.run(fmax=0.01)

# Final end-point:
qn = CatLearnMin(slab_final, trajectory='final.traj')
qn.run(fmax=0.01)

# Set number of images
n_images = 7

# 2.A. NEB using ASE #########################################################

initial_ase = read('initial.traj')
final_ase = read('final.traj')

ase_calculator = copy.deepcopy(ase_calculator)

images_ase = [initial_ase]
for i in range(1, n_images-1):
    image = initial_ase.copy()
    image.set_calculator(copy.deepcopy(ase_calculator))
    images_ase.append(image)

images_ase.append(final_ase)

neb_ase = NEB(images_ase, climb=True)
neb_ase.interpolate(method='idpp')

qn_ase = MDMin(neb_ase, trajectory='neb_ase.traj')
qn_ase.run(fmax=0.05)

nebtools_ase = NEBTools(images_ase)

Sf_ase = nebtools_ase.get_fit()[2]
Ef_ase = nebtools_ase.get_fit()[3]

Ef_neb_ase, dE_neb_ase = nebtools_ase.get_barrier(fit=False)
nebtools_ase.plot_band()
plt.show()

# 2.B. NEB using CatLearn ####################################################

neb_catlearn = CatLearnNEB(start='initial.traj', end='final.traj',
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
