from ase.calculators.emt import EMT
from ase.io import read, write
from ase.optimize import BFGS
import copy
import glob
import shutil
import os
from catlearn.optimize.autoneb_ase import AutoNEBASE
from catlearn.optimize.catlearn_autoneb_optimizer import AutoNEBOptimizer

""" 
    Toy model rearrangement of Pt heptamer island on Pt(111).
    This example contains: 
    1) Optimization of the initial and final end-points of the reaction path. 
    2A) NEB optimization using AutoNEB as implemented in ASE. 
    2B) NEB optimization using our machine-learning surrogate model.
    3) Comparison between the ASE NEB and our Machine Learning assisted NEB 
       algorithm.
"""

# 0. Clean up before running. ################################################

# Check there are no previous images:
for i in glob.glob("images*"):
    os.remove(i)

# Remove data from previous iteration:
directory = './AutoNEB_iter'
if os.path.exists(directory):
    shutil.rmtree(directory)

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
qn = BFGS(slab_initial, trajectory='initial.traj')
qn.run(fmax=0.01)

# Final end-point:
qn = BFGS(slab_final, trajectory='final.traj')
qn.run(fmax=0.01)

# Set number of images
n_images = 7

# 2.A. AutoNEB using ASE  ####################################################

# Write in format for AutoNEB:
initial = read('initial.traj')
write('images000.traj', initial)

final = read('final.traj')
write('images001.traj', final)

automatic = AutoNEBASE(prefix='images',
                       n_max=n_images,
                       n_simul=1,
                       attach_calculators=ase_calculator)
automatic.run()

# 2.B. AutoNEB using CatLearn ################################################

auto_catlearn = AutoNEBOptimizer(start='initial.traj',
                                 end='final.traj',
                                 ase_calc=copy.deepcopy(ase_calculator),
                                 n_images=n_images)
auto_catlearn.run(fmax=0.05, plot_neb_paths=True)
