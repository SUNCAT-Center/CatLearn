from catlearn.optimize.catlearn_minimizer import CatLearnMinimizer
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

catlearn_version = '_u_1_2_0'

results_dir = './Results/'

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# 0. Set calculator.
calc = Himmelblau()

# 1. Set common initial structure.
mol = Atoms('C', positions=[(-1.0, -1.0, 0.0)])
mol.rattle(stdev=0.1, seed=0)

# 2. Benchmark.
minimizers = ['BFGS', 'LBFGS', 'FIRE']

for i in minimizers:
    filename = i + '_opt.traj'
    if not os.path.exists(results_dir + filename):
        initial = mol.copy()
        initial.set_calculator(calc)
        opt = eval(i)(initial, trajectory=filename, )
        opt.run(fmax=0.05, steps=500)
        shutil.copy('./' + filename, results_dir + filename)

minimizers = ['BFGS', 'FIRE']

for i in minimizers:
    filename = i + '_opt' + catlearn_version
    if not os.path.exists(results_dir + filename + '_catlearn.traj'):
        initial = mol.copy()
        initial.set_calculator(calc)
        opt = CatLearnMinimizer(initial, filename=filename)
        opt.run(fmax=0.05, ml_algo=i)
        shutil.copy('./' + filename + '_catlearn.traj',
                    results_dir + filename + '_catlearn.traj')


# # 3. Summary of the results:
print('\n Summary of the results:\n ------------------------------------')

# Function evaluations:
for filename in os.listdir(results_dir):
    atoms = read(filename,':')
    feval = len(atoms)
    print('Number of function evaluations using ' + filename+ ':', feval)