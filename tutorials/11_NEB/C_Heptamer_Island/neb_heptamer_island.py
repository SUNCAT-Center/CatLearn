from ase.build import fcc100, add_adsorbate
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.optimize import QuasiNewton
from ase.io import read
from ase.constraints import FixAtoms
from ase.neb import NEB
from ase.optimize import LBFGS, FIRE, BFGS, BFGSLineSearch, MDMin
from ase.visualize import view
import matplotlib.pyplot as plt
from catlearn.optimize.catlearn_neb_optimizer import NEBOptimizer
from ase.neb import NEBTools
import copy

# 1. Structural relaxation. ##################################################

# Setup calculator:
ase_calculator = EMT()

# # 1.1. Structures:

slab_initial = read('./A_structure/POSCAR')
# slab_initial = read('./translate_H_initial/POSCAR')
slab_initial.set_calculator(copy.deepcopy(ase_calculator))

slab_final = read('./I_structure/POSCAR')
# slab_final = read('./translate_H_final/POSCAR')
slab_final.set_calculator(ase_calculator)


# 1.2. Optimize initial and final end-points.

# Initial end-point:
qn = FIRE(slab_initial, trajectory='initial.traj')
qn.run(fmax=0.01)

# Final end-point:
qn = FIRE(slab_final, trajectory='final.traj')
qn.run(fmax=0.01)


# 2.A. NEB using ASE #########################################################

# initial_ase = read('initial.traj')
# final_ase = read('final.traj')
#
# ase_calculator = copy.deepcopy(ase_calculator)
#
# n_images = 7
# images_ase = [initial_ase]
# for i in range(n_images):
#     image = initial_ase.copy()
#     image.set_calculator(copy.deepcopy(ase_calculator))
#     images_ase.append(image)
#
# images_ase.append(final_ase)
#
# neb_ase = NEB(images_ase, climb=True, method='improvedtangent', k=0.1)
# neb_ase.interpolate(method='idpp')
# qn_ase = FIRE(neb_ase, trajectory='neb_ase.traj')
# qn_ase.run(fmax=0.05)
# nebtools_ase = NEBTools(images_ase)
#
# Sf_ase = nebtools_ase.get_fit()[2]
# Ef_ase = nebtools_ase.get_fit()[3]
#
# Ef_neb_ase, dE_neb_ase = nebtools_ase.get_barrier(fit=False)
# nebtools_ase.plot_band()
#
# view(slab_initial)
# view(slab_final)
# final_neb_ase = read('neb_ase.traj',':')
# view(final_neb_ase)
#
# plt.show()

# 2.B. NEB using CatLearn ####################################################

neb_catlearn = NEBOptimizer(start='initial.traj', end='final.traj',
                       ase_calc=copy.deepcopy(ase_calculator), n_images=6,
                       interpolation='idpp')

neb_catlearn.run(max_iter=200, ml_algo='FIRE', climb_img=True, max_step=0.10,
                 neb_method='improvedtangent', store_neb_paths=True)