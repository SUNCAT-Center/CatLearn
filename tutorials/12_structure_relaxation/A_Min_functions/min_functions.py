from catlearn.optimize.catlearn_minimizer import CatLearnMin
from catlearn.optimize.functions_calc import Himmelblau, NoiseHimmelblau, \
GoldsteinPrice, Rosenbrock, MullerBrown
from ase import Atoms
from ase.optimize import *
from ase.optimize.sciopt import *
from ase.io import read
import copy
import numpy as np
import os
import shutil

""" 
    Minimization example.
    Toy model using toy model potentials such as Muller-Brown, Himmelblau, 
    Goldstein-Price or Rosenbrock.       
"""

# 0. Set calculator.
calculator = NoiseHimmelblau()

# 1. Set common initial structure.
initial_structure = Atoms('H', positions=[(1.0, 1.0, 0.0)])
initial_structure.rattle(seed=0, stdev=0.1)
# 2. Benchmark.
# 2.A. Optimize structure using CatLearn:
initial_catlearn = initial_structure.copy()
initial_catlearn.set_calculator(calculator)

catlearn_opt = CatLearnMin(initial_catlearn, trajectory='catlearn_opt.traj')
catlearn_opt.run(fmax=0.05)

# 2.B. Optimize structure using ASE.
initial_gpmin = initial_structure.copy()
initial_gpmin.set_calculator(calculator)

gpmin_opt = GPMin(initial_gpmin, trajectory='gpmin_opt.traj',
                 update_hyperparams=True)
gpmin_opt.run(fmax=0.05, steps=200)

# 2.C. Optimize structure using BFGS.
initial_bfgs = initial_structure.copy()
initial_bfgs.set_calculator(calculator)

bfgs_opt = SciPyFminBFGS(initial_bfgs, trajectory='bfgs_opt.traj')
bfgs_opt.run(fmax=0.05, steps=200)
print(bfgs_opt.force_calls)
print(bfgs_opt.__dict__)

# 2.D. Optimize structure using BFGS.
initial_fire = initial_structure.copy()
initial_fire.set_calculator(calculator)

fire_opt = FIRE(initial_fire, trajectory='fire_opt.traj')
fire_opt.run(fmax=0.05, steps=200)

# 3. Summary of the results:
print('\n Summary of the results:\n ------------------------------------')

catlearn_results = read('catlearn_opt.traj', ':')
print('Number of function evaluations using CatLearn:', len(catlearn_results))

gpmin_results = read('gpmin_opt.traj', ':')
print('Number of function evaluations using GPMin:', gpmin_opt.function_calls)

bfgs_results = read('bfgs_opt.traj', ':')
print('Number of function evaluations using BFGS:', len(bfgs_results))

fire_results = read('fire_opt.traj', ':')
print('Number of function evaluations using FIRE:', len(fire_results))


print('Energy CatLearn:', catlearn_results[-1].get_potential_energy())
print('Energy GPMin:', gpmin_results[-1].get_potential_energy())
print('Energy BFGS:', bfgs_results[-1].get_potential_energy())
print('Energy FIRE:', fire_results[-1].get_potential_energy())