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

catlearn_version = '1_0_0'

# 0. Set calculator.
calculator = NoiseHimmelblau()

# 1. Set common initial structure.
initial_structure = Atoms('C', positions=[(-1.0, -1.0, 0.0)])

# 2. Benchmark.
# 2.A. Optimize structure using CatLearn:
initial_catlearn = initial_structure.copy()
initial_catlearn.set_calculator(calculator)

catlearn_opt = CatLearnMin(initial_catlearn, trajectory='catlearn_opt.traj')
catlearn_opt.run(fmax=0.05)

# 2.B. Optimize structure using ASE.
initial_ase = initial_structure.copy()
initial_ase.set_calculator(calculator)

ase_opt = BFGS(initial_ase, trajectory='ase_opt.traj')
ase_opt.run(fmax=0.05, steps=500)

# 3. Summary of the results:
print('\n Summary of the results:\n ------------------------------------')

catlearn_results = read('catlearn_opt.traj', ':')
print('Number of function evaluations using CatLearn:', len(catlearn_results))

ase_results = read('ase_opt.traj', ':')
print('Number of function evaluations using ASE:', len(ase_results))

print('Energy CatLearn:', catlearn_results[-1].get_potential_energy())
print('Energy ASE:', ase_results[-1].get_potential_energy())