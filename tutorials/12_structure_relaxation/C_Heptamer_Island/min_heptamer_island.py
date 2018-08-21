from catlearn.optimize.catlearn_minimizer import CatLearnMinimizer
from ase.calculators.emt import EMT
from ase.io import read
from ase.optimize import BFGS
import copy
from ase.visualize import view
import numpy as np


""" 
    Toy model minimization of Pt heptamer island on Pt(111).
    Minimization example. 
"""

# 1. Structural relaxation. ##################################################

# Setup calculator:
ase_calculator = EMT()

# 1.1. Structures:

common_initial = read('./A_structure/POSCAR')

np.random.seed(1)
for i in common_initial:
    if i.position[2] > 14.00:
        i.symbol = 'Au'
        i.position = i.position + np.random.normal(scale=0.2)

# 2.A. Optimize structure using ASE.
initial_ase = copy.deepcopy(common_initial)
initial_ase.set_calculator(ase_calculator)

ase_opt = BFGS(initial_ase, trajectory='ase_optimization.traj')
ase_opt.run(fmax=0.01)

# 2.B. Optimize structure using CatLearn:

initial_catlearn = copy.deepcopy(common_initial)
initial_catlearn.set_calculator(ase_calculator)

catlearn_opt = CatLearnMinimizer(initial_catlearn, filename='results')
catlearn_opt.run(fmax=0.01)

# 3. Summary of the results:
print('\n Summary of the results:\n ------------------------------------')

ase_results = read('ase_optimization.traj', ':')

print('Number of function evaluations using ASE:', len(ase_results))
print('Energy ASE (eV):', ase_results[-1].get_potential_energy())

catlearn_results = read('results_catlearn.traj', ':')

print('Number of function evaluations using CatLearn:', len(catlearn_results))
print('Energy CatLearn (eV):', catlearn_results[-1].get_potential_energy())
