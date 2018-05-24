from ase.build import fcc100, add_adsorbate
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.optimize import QuasiNewton
from ase.io import read
from ase.constraints import FixAtoms
from ase.neb import NEB
from ase.optimize import BFGS
from ase.visualize import view
import matplotlib.pyplot as plt
from catlearn.optimize.catlearn_neb_optimizer import NEBOptimizer
from ase.neb import NEBTools
import copy
from catlearn.optimize.mullerbrown_calc import MullerBrown
from ase import Atoms
from ase.optimize import LBFGS, FIRE, BFGS, BFGSLineSearch, MDMin
import numpy as np

""" Toy model using the MullerBrown potential.  
In this tutorial, we first optimize the initial and final end-points of the 
reaction path. Secondly, we make a comparission between the ASE 
implementation and our CatLearn (Machine Learning assisted) NEB code.
"""


# 1. Structural relaxation. ##################################################

# 1.1. Structures:
initial_structure = Atoms('C', positions=[(-0.55, 1.41, 0.0)])
final_structure = Atoms('C', positions=[(0.625, 0.025, 0.0)])

initial_structure.set_calculator(MullerBrown())
final_structure.set_calculator(MullerBrown())

# 1.2. Optimize initial and final states.

# Initial:
initial_opt = FIRE(initial_structure, trajectory='initial_optimized.traj')
initial_opt.run(fmax=0.01)
# Final:
final_opt = FIRE(final_structure, trajectory='final_optimized.traj')
final_opt.run(fmax=0.01)

# 2.A. NEB using ASE #########################################################

number_of_images_neb_exact = 7
initial_exact = read('initial_optimized.traj')
final_exact = read('final_optimized.traj')
images_exact = [initial_exact]
for i in range(0,number_of_images_neb_exact):
    image_exact = initial_exact.copy()
    image_exact.set_calculator(MullerBrown())
    images_exact.append(image_exact)
images_exact.append(final_exact)

neb_exact = NEB(images_exact, climb=True, method='improvedtangent', k=0.1)
neb_exact.interpolate()

qn_exact = FIRE(neb_exact, trajectory='neb_exact.traj')
qn_exact.run(fmax=0.01)

nebtools_exact = NEBTools(images_exact)

Sf_exact = nebtools_exact.get_fit()[2]
Ef_exact = nebtools_exact.get_fit()[3]

Ef_neb_exact, dE_neb_exact = nebtools_exact.get_barrier(fit=False)
nebtools_exact.plot_band()
plt.show()


# 2.B. NEB using CatLearn ####################################################

initial = read('initial_optimized.traj')
final = read('final_optimized.traj')

neb_catlearn = NEBOptimizer(start='initial_optimized.traj',
                       end='final_optimized.traj', ase_calc=MullerBrown(),
                       n_images=9, interpolation='')

neb_catlearn.run(ml_algo='FIRE', k=100.0, climb_img=True, max_step=0.05,
neb_method='improvedtangent', store_neb_paths=True)

# 3. Summary of the results #################################################
