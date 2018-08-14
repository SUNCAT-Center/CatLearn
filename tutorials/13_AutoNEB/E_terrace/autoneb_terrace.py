from ase.build import fcc100, add_adsorbate
from ase.calculators.emt import EMT
from ase.io import read
from ase.constraints import FixAtoms
from ase.neb import NEB
from ase.optimize import BFGS, MDMin, FIRE
import matplotlib.pyplot as plt
from catlearn.optimize.catlearn_neb_optimizer import CatLearnNEB
from catlearn.optimize.catlearn_autoneb_optimizer import CatLearnAutoNEB

from ase.neb import NEBTools
import copy
from ase.build import surface
from ase.visualize import view
from ase.build import add_adsorbate

""" 
    Toy model for the diffusion of a Pt atom on an Pt(211) surface.  
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

slab = read('initial.traj')
slab.set_calculator(copy.deepcopy(ase_calculator))
qn = BFGS(slab, trajectory='initial_opt.traj')
qn.run(fmax=0.01)

# Final end-point:
slab = read('final.traj')
slab.set_calculator(copy.deepcopy(ase_calculator))
qn = BFGS(slab, trajectory='final_opt.traj')
qn.run(fmax=0.01)


# # Define number of images:
n_images = 7

# 2.A. NEB using ASE #########################################################

initial_ase = read('initial_opt.traj')
final_ase = read('final_opt.traj')

images_ase = [initial_ase]
for i in range(1, n_images-1):
    image = initial_ase.copy()
    image.set_calculator(copy.deepcopy(ase_calculator))
    images_ase.append(image)
images_ase.append(final_ase)

neb_ase = NEB(images_ase, climb=True, method='aseneb')
neb_ase.interpolate(method='idpp')

qn_ase = MDMin(neb_ase, trajectory='neb_ase.traj')
qn_ase.run(fmax=0.05)

nebtools_ase = NEBTools(images_ase)

Sf_ase = nebtools_ase.get_fit()[2]
Ef_ase = nebtools_ase.get_fit()[3]

Ef_neb_ase, dE_neb_ase = nebtools_ase.get_barrier(fit=False)
nebtools_ase.plot_band()

plt.show()


neb_catlearn = CatLearnAutoNEB(start='initial_opt.traj',
                           end='final_opt.traj',
                           ase_calc=copy.deepcopy(ase_calculator),
                           n_images=n_images,
                           restart=False)

neb_catlearn.run(fmax=0.05, plot_neb_paths=True, acquisition='acq_1',
                 )