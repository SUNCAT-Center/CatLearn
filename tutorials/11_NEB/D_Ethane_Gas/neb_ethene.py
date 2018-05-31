from ase.build import molecule
from ase.neb import NEB, NEBTools
from ase.calculators.emt import EMT
from ase.optimize.fire import FIRE
from ase.optimize.mdmin import MDMin
import copy
from ase.visualize import view
from ase.io import read, write
import matplotlib.pyplot as plt
from catlearn.optimize.catlearn_neb_optimizer import NEBOptimizer


ase_calculator = EMT()


# 0. Build and optimize end-points. #########################################

# 0.1 Optimise initial state.
initial = molecule('C2H6')
initial.set_calculator(copy.deepcopy(ase_calculator))
qn = FIRE(initial, trajectory='initial.traj')
qn.run(fmax=0.01)

# 0.2 Create and optimize final state.
final = initial.copy()
final.positions[2:5] = initial.positions[[3, 4, 2]]
final.set_calculator(copy.deepcopy(ase_calculator))
qn = FIRE(final, trajectory='final.traj')
qn.run(fmax=0.01)


# 1. NEB using CatLearn ####################################################

# 1.1 Generate blank images (total of 9 copies + 2 end-points = 11 images)
images = [initial]
for i in range(27):
    images.append(initial.copy())
for image in images:
    image.set_calculator(copy.deepcopy(ase_calculator))
images.append(final)

# 1.2 Run IDPP interpolation
ase_neb = NEB(images, climb=False)
ase_neb.interpolate('linear')

# 1.3 Run NEB calculation
qn = FIRE(ase_neb, trajectory='ethane_linear.traj')
qn.run(fmax=0.05)

# 1.4 Plot converged NEB path.
nebtools = NEBTools(images)
fig = nebtools.plot_band()
plt.show()


# 2.B. NEB using CatLearn ####################################################


neb_catlearn = NEBOptimizer(start='initial.traj', end='final.traj',
                       ase_calc=copy.deepcopy(ase_calculator), n_images=29,
                       interpolation='linear')

neb_catlearn.run(ml_algo='FIRE', climb_img=True, max_step=0.05,
                 neb_method='improvedtangent', plot_neb_paths=True)








