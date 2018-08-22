from ase import Atoms
from gpaw import GPAW
from ase.optimize import BFGS, FIRE, MDMin
from ase.optimize.sciopt import SciPyFminPowell, SciPyFminBFGS
from catlearn.optimize.catlearn_minimizer import CatLearnMinimizer
from ase.io import read
import copy
from ase.db import connect
import ase.db


# 0. Setup calculator:
###############################################################################
ase_calculator = GPAW(mode='lcao',
     basis='dzp',
     kpts={'density': 2.0})

# 1 Structure:
db = ase.db.connect('systems.db')
common_initial = db.get_atoms('Au8CO')


np.random.seed(1)
for i in common_initial:
    if i.position[2] > 8.50:
        i.position = i.position + np.random.normal(scale=0.0)

# 2.A. Optimize structure using BFGS.
###############################################################################

initial_ase = copy.deepcopy(common_initial)
initial_ase.set_calculator(ase_calculator)

bfgs_opt = BFGS(initial_ase, trajectory='bfgs_optimization.traj')
bfgs_opt.run(fmax=0.02)

# 2.B. Optimize structure using CatLearn:
###############################################################################

initial_catlearn = copy.deepcopy(common_initial)
initial_catlearn.set_calculator(ase_calculator)

catlearn_opt = CatLearnMinimizer(initial_catlearn, filename='results')
catlearn_opt.run(fmax=0.02, ml_algo='SciPyFminCG')

# 2.C. Optimize structure using FIRE:
###############################################################################

initial_fire = copy.deepcopy(common_initial)
initial_fire.set_calculator(ase_calculator)

fire_opt = FIRE(initial_fire, trajectory='fire_optimization.traj')
fire_opt.run(fmax=0.02)

# 2.D. Optimize structure using MDMin:
###############################################################################

initial_mdmin = copy.deepcopy(common_initial)
initial_mdmin.set_calculator(ase_calculator)

mdmin_opt = MDMin(initial_mdmin, trajectory='mdmin_optimization.traj')
mdmin_opt.run(fmax=0.02)

# 2.E. Optimize structure using SciPyFminCG:
###############################################################################

# initial_scipy = copy.deepcopy(common_initial)
# initial_scipy.set_calculator(ase_calculator)
#
# scipy_opt = SciPyFminBFGS(initial_scipy, trajectory='scipy_optimization.traj')
# scipy_opt.run(fmax=0.02)


# 3. Summary of the results:
###############################################################################

print('\n Summary of the results:\n ------------------------------------')

bfgs_results = read('bfgs_optimization.traj', ':')

print('Number of function evaluations using BFGS:', len(bfgs_results))
print('Energy BFGS (eV):', bfgs_results[-1].get_potential_energy())

fire_results = read('fire_optimization.traj', ':')

print('Number of function evaluations using FIRE:', len(fire_results))
print('Energy FIRE (eV):', fire_results[-1].get_potential_energy())

mdmin_results = read('mdmin_optimization.traj', ':')

print('Number of function evaluations using MDMin:', len(mdmin_results))
print('Energy MDMin (eV):', mdmin_results[-1].get_potential_energy())


catlearn_results = read('results_catlearn.traj', ':')

print('Number of function evaluations using CatLearn:', len(catlearn_results))
print('Energy CatLearn (eV):', catlearn_results[-1].get_potential_energy())

# scipy_results = read('scipy_optimization.traj', ':')
#
# print('Number of function evaluations using SciPy:', len(scipy_results))
# print('Energy SciPy (eV):', scipy_results[-1].get_potential_energy())


