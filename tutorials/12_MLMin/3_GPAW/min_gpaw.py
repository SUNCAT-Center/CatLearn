from ase import Atoms
from gpaw import GPAW
from ase.optimize import *
from catlearn.optimize.mlmin import MLMin
from ase.io import read
from ase.constraints import FixAtoms

""" 
    Structure relaxation H2O using GPAW.
    Benchmark using MLMin, GPMin, LBFGS and FIRE. 
"""

# 1. Build Atoms Object.
###############################################################################

# Setup calculator:

calc = GPAW(mode='lcao',
            basis='dzp',
            kpts={'density': 4.0})

# 1.1. Set up structure:
a = 6
b = a / 2
atoms = Atoms('H2O',
            [(b, 0.7633 + b, -0.4876 + b),
             (b, -0.7633 + b, -0.4876 + b),
             (b, b, 0.1219 + b)], cell=[a, a, a])
atoms.rattle(stdev=0.1, seed=0)

c = FixAtoms(indices=[0])
atoms.set_constraint(c)


# 2. Benchmark.
###############################################################################

# 2.A. Optimize structure using MLMin (CatLearn).
initial_mlmin = atoms.copy()
initial_mlmin.set_calculator(calc)
mlmin_opt = MLMin(initial_mlmin, trajectory='results_catlearn.traj')
mlmin_opt.run(fmax=0.01, kernel='SQE')

# 2.B Optimize using GPMin.
initial_gpmin = atoms.copy()
initial_gpmin.set_calculator(calc)
gpmin_opt = GPMin(initial_gpmin, trajectory='results_gpmin.traj',
                  update_hyperparams=True)
gpmin_opt.run(fmax=0.01)

# 2.C Optimize using LBFGS.
initial_lbfgs = atoms.copy()
initial_lbfgs.set_calculator(calc)
lbfgs_opt = LBFGS(initial_lbfgs, trajectory='results_lbfgs.traj')
lbfgs_opt.run(fmax=0.01)

# 2.D Optimize using FIRE.
initial_fire = atoms.copy()
initial_fire.set_calculator(calc)
fire_opt = FIRE(initial_fire, trajectory='results_fire.traj')
fire_opt.run(fmax=0.01)

# 3. Summary of the results:
###############################################################################

print('\n Summary of the results:\n ------------------------------------')

fire_results = read('results_fire.traj', ':')
print('Number of function evaluations using FIRE:',
      len(fire_results))

lbfgs_results = read('results_lbfgs.traj', ':')
print('Number of function evaluations using LBFGS:',
      len(lbfgs_results))

gpmin_results = read('results_gpmin.traj', ':')
print('Number of function evaluations using GPMin:', gpmin_opt.function_calls)

catlearn_results = read('results_catlearn.traj', ':')
print('Number of function evaluations using MLMin (CatLearn):',
      len(catlearn_results))
