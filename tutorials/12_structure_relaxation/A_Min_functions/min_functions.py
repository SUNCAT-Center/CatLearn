from catlearn.optimize.catlearn_minimizer import CatLearnMinimizer
from catlearn.optimize.functions_calc import Himmelblau, NoiseHimmelblau, \
GoldsteinPrice, Rosenbrock, MullerBrown
from ase import Atoms
from ase.optimize import BFGS
from ase.io import read
import copy
import numpy as np

""" 
    Minimization example.
    Toy model using toy model potentials such as Muller-Brown, Himmelblau, 
    Goldstein-Price or Rosenbrock.       
"""

# 0. Set calculator.
ase_calculator = MullerBrown()

# 1. Set common initial structure.
common_initial = Atoms('C', positions=[(-1.5, -1.0, 0.0)])
common_initial.rattle(stdev=0.5, seed=0)

# 2.A. Optimize structure using ASE.
initial_ase = copy.deepcopy(common_initial)
initial_ase.set_calculator(copy.deepcopy(ase_calculator))

ase_opt = BFGS(initial_ase, trajectory='ase_optimization.traj')
ase_opt.run(fmax=0.01, steps=500)

# 2.B. Optimize structure using CatLearn:

initial_catlearn = copy.deepcopy(common_initial)
initial_catlearn.set_calculator(copy.deepcopy(ase_calculator))

catlearn_opt = CatLearnMinimizer(initial_catlearn, filename='results')
catlearn_opt.run(fmax=0.01)

# 3. Summary of the results:
print('\n Summary of the results:\n ------------------------------------')

ase_results = read('ase_optimization.traj', ':')

print('Number of function evaluations using ASE:', len(ase_results))
print('Energy ASE (eV):', ase_results[-1].get_potential_energy())
print('Coordinates ASE:', ase_results[-1].get_positions().flatten())

catlearn_results = read('results_catlearn.traj', ':')

print('Number of function evaluations using CatLearn:', len(catlearn_results))
print('Energy CatLearn (eV):', catlearn_results[-1].get_potential_energy())
print('Coordinates CatLearn:', catlearn_results[-1].get_positions().flatten())