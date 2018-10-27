from ase.io import read
from ase.neb import NEB
import matplotlib.pyplot as plt
from catlearn.optimize.catlearn_neb_optimizer import CatLearnNEB
from ase.neb import NEBTools
import copy
from catlearn.optimize.functions_calc import Himmelblau
from ase import Atoms
from ase.optimize import BFGS, MDMin

""" 
    Toy model using the Himmelblau potential.  
    This example contains: 
    1) Optimization of the initial and final end-points of the reaction path. 
    2A) NEB optimization using NEB-CI as implemented in ASE. 
    2B) NEB optimization using our machine-learning surrogate model.
    3) Comparison between the ASE NEB and our Machine Learning assisted NEB 
       algorithm.
"""

# 1. Structural relaxation. ##################################################

# Setup calculator:
ase_calculator = Himmelblau()

# # 1.1. Structures:
initial_structure = Atoms('C', positions=[(-3., -3., 0.)])
final_structure = Atoms('C', positions=[(3., 3., 0.)])

initial_structure.set_calculator(copy.deepcopy(ase_calculator))
final_structure.set_calculator(copy.deepcopy(ase_calculator))

# 1.2. Optimize initial and final end-points.

# Initial end-point:
initial_opt = BFGS(initial_structure, trajectory='initial_optimized.traj')
initial_opt.run(fmax=0.01)

# Final end-point:
final_opt = BFGS(final_structure, trajectory='final_optimized.traj')
final_opt.run(fmax=0.01)

# # Define number of images for the NEB:

n_images = 7

# 2.A. NEB using ASE #########################################################

initial_ase = read('initial_optimized.traj')
final_ase = read('final_optimized.traj')
images_ase = [initial_ase]
for i in range(1, n_images-1):
    image_ase = initial_ase.copy()
    image_ase.set_calculator(copy.deepcopy(ase_calculator))
    images_ase.append(image_ase)
images_ase.append(final_ase)

neb_ase = NEB(images_ase, climb=True, method='improvedtangent')
neb_ase.interpolate()
qn_ase = MDMin(neb_ase, trajectory='neb_ase.traj')
qn_ase.run(fmax=0.01)

nebtools_ase = NEBTools(images_ase)

Sf_ase = nebtools_ase.get_fit()[2]
Ef_ase = nebtools_ase.get_fit()[3]

Ef_neb_ase, dE_neb_ase = nebtools_ase.get_barrier(fit=False)
nebtools_ase.plot_band()

plt.show()

# 2.B. NEB using CatLearn ####################################################

initial = read('initial_optimized.traj')
final = read('final_optimized.traj')

neb_catlearn = CatLearnNEB(start='initial_optimized.traj',
                           end='final_optimized.traj',
                           ase_calc=copy.deepcopy(ase_calculator),
                           n_images=7,
                           interpolation='linear', restart=False)

neb_catlearn.run(fmax=0.05, plot_neb_paths=True, acquisition='acq_2')

# 3. Summary of the results #################################################

# NEB ASE:
print('\nSummary of the results: ', '\n------------------------\n')

atoms_ase = read('neb_ase.traj', ':')
n_eval_ase = len(atoms_ase) - 2 * n_images

print('Number of function evaluations CI-NEB implemented in ASE:', n_eval_ase)

# CatLearn:
atoms_catlearn = read('evaluated_structures.traj', ':')
n_eval_catlearn = len(atoms_catlearn) - 2
print('Number of function evaluations CatLearn:', n_eval_catlearn)

# Comparison:
print('\nThe CatLearn algorithm required ',
      (n_eval_ase/n_eval_catlearn),
      'times less number of function evaluations than '
      'the standard NEB algorithm.')
