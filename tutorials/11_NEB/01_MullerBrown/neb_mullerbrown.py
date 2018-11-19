from ase.io import read
from ase.neb import NEB
import matplotlib.pyplot as plt
from catlearn.optimize.mlneb import MLNEB
from ase.neb import NEBTools
import copy
from catlearn.optimize.functions_calc import MullerBrown
from ase import Atoms
from ase.optimize import BFGS, MDMin
from catlearn.optimize.tools import plotneb


""" 
    Toy model using the MullerBrown potential.  
    This example contains: 
    1. Optimization of the initial and final end-points of the reaction path. 
    2.A. NEB optimization using CI-NEB as implemented in ASE. 
    2.B. NEB optimization using our machine-learning surrogate model.
    3. Comparison between the ASE NEB and our ML-NEB algorithm.
"""

# 1. Structural relaxation. ##################################################

# Setup calculator:
ase_calculator = MullerBrown()

# # 1.1. Structures:
initial_structure = Atoms('C', positions=[(-0.55, 1.41, 0.0)])
final_structure = Atoms('C', positions=[(0.626, 0.025, 0.0)])

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

n_images = 11

# 2.A. NEB using ASE

initial_ase = read('initial_optimized.traj')
final_ase = read('final_optimized.traj')
images_ase = [initial_ase]
for i in range(1, n_images-1):
    image_ase = initial_ase.copy()
    image_ase.set_calculator(copy.deepcopy(ase_calculator))
    images_ase.append(image_ase)
images_ase.append(final_ase)

neb_ase = NEB(images_ase, climb=True)
neb_ase.interpolate(method='idpp')
qn_ase = MDMin(neb_ase, trajectory='neb_ase.traj')
qn_ase.run(fmax=0.05)

# 2.B. ML-NEB using CatLearn

initial = read('initial_optimized.traj')
final = read('final_optimized.traj')

neb_catlearn = MLNEB(start='initial_optimized.traj',
                     end='final_optimized.traj',
                     ase_calc=copy.deepcopy(ase_calculator),
                     n_images=n_images,
                     interpolation='linear', restart=False)

neb_catlearn.run(fmax=0.05, trajectory='ML-NEB.traj')

# 3. Summary of the results #################################################

# NEB ASE:
print('\nSummary of the results: \n')

atoms_ase = read('neb_ase.traj', ':')
n_eval_ase = int(len(atoms_ase) - 2 * (len(atoms_ase)/n_images))

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
