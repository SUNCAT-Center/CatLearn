from ase.calculators.emt import EMT
from ase.io import read
from ase.optimize import BFGS, FIRE, MDMin, GPMin
from ase.optimize.sciopt import *
from catlearn.optimize.catlearn_minimizer import CatLearnMin
from ase.visualize import view

""" 
    CatLearn Minimizer. 
    Example 1.
    Optimization of Au island on Pt(111).     
"""


# Setup calculator:
calculator = EMT()

# 1.1. Structures:

initial_structure = read('preoptimized_structure.traj')
np.random.seed(5)
for i in initial_structure:
    if i.position[2] > 14.00:
        i.symbol = 'Au'
        i.position = i.position + np.random.normal(scale=0.1)

# 2.A. Optimize structure using CatLearn:

initial_catlearn = initial_structure.copy()
initial_catlearn.set_calculator(calculator)

catlearn_opt = CatLearnMin(initial_catlearn, trajectory='catlearn_opt.traj')
catlearn_opt.run(fmax=0.05)

# 2.B. Optimize structure using ASE.
initial_ase = initial_structure.copy()
initial_ase.set_calculator(calculator)

ase_opt = GPMin(initial_ase, trajectory='ase_opt.traj',
                update_hyperparams=True)
ase_opt.run(fmax=0.05)


# 3. Summary of the results:
print('\n Summary of the results:\n ------------------------------------')

catlearn_results = read('catlearn_opt.traj', ':')
print('Number of function evaluations using CatLearn:', len(catlearn_results))

ase_results = read('ase_opt.traj', ':')
print('Number of function evaluations using ASE:', len(ase_results))

