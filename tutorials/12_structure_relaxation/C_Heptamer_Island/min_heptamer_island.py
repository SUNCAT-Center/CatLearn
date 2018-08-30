from catlearn.optimize.catlearn_minimizer import CatLearnMinimizer
from ase.calculators.emt import EMT
from ase.io import read
from ase.optimize import *
from ase.optimize.sciopt import *
import copy
from ase.visualize import view
import numpy as np
import os, shutil

""" 
    Toy model minimization of Pt heptamer island on Pt(111).
    Minimization example. 
"""

catlearn_version = '_u_1_7_0'

system = '_rattle01'

# 1. Structural relaxation. ##################################################

# Setup calculator:
calc = EMT()

# 1.1. Structures:

mol = read('./A_structure/POSCAR')

np.random.seed(1)
for i in mol:
    if i.position[2] > 14.00:
        i.symbol = 'Au'
        i.position = i.position + np.random.normal(scale=0.2)

# 3. Benchmark.
###############################################################################

# 2.A. Optimize structure using ASE.
initial_catlearn = mol.copy()
initial_catlearn.set_calculator(calc)
ase_opt = CatLearnMinimizer(initial_catlearn,
                            filename='results',
                            ml_calc='SQE_sequential')
ase_opt.run(fmax=0.01, ml_algo='BFGS')


# 2.B Optimize using GPMin.
initial_gpmin = mol.copy()
initial_gpmin.set_calculator(calc)
gpmin_opt = GPMin(initial_gpmin, trajectory='results_gpmin.traj')
gpmin_opt.run(fmax=0.01)


# 3. Summary of the results:
print('\n Summary of the results:\n ------------------------------------')

gpmin_results = read('results_gpmin.traj', ':')

print('Number of function evaluations using ASE:', len(gpmin_results))
print('Energy GPMin (eV):', gpmin_results[-1].get_potential_energy())

catlearn_results = read('results_catlearn.traj', ':')

print('Number of function evaluations using CatLearn:', len(catlearn_results))
print('Energy CatLearn (eV):', catlearn_results[-1].get_potential_energy())
