from gpaw import GPAW
import ase.db
from catlearn.optimize.catlearn_minimizer import CatLearnMin
import numpy as np
from ase.visualize import view
from ase.optimize import BFGS, FIRE, MDMin, GPMin
from ase.optimize.sciopt import *
from ase.io import read
""" 
    CatLearn Minimizer. 
    Example 3.
    Optimization of CO on Au(111) using GPAW.     
"""

# Setup calculator:

calculator = GPAW(mode='lcao',
                  basis='dzp',
                  kpts={'density': 2.0})

# 1.1. Structures:
db = ase.db.connect('systems.db')
initial_structure = db.get_atoms(formula='Au8CO')

np.random.seed(2)
for i in initial_structure:
    if i.position[2] > 8.50:
        i.position = i.position + np.random.normal(scale=0.2)

# 2.A. Optimize structure using CatLearn:
initial_catlearn = initial_structure.copy()
initial_catlearn.set_calculator(calculator)

catlearn_opt = CatLearnMin(initial_catlearn, trajectory='catlearn_opt.traj')
catlearn_opt.run(fmax=0.05)

# 2.B. Optimize structure using ASE.
initial_ase = initial_structure.copy()
initial_ase.set_calculator(calculator)

ase_opt = GPMin(initial_ase, trajectory='ase_opt.traj')
ase_opt.run(fmax=0.05)

# 3. Summary of the results:
print('\n Summary of the results:\n ------------------------------------')

catlearn_results = read('catlearn_opt.traj', ':')
print('Number of function evaluations using CatLearn:', len(catlearn_results))

ase_results = read('ase_opt.traj', ':')
print('Number of function evaluations using ASE:', len(ase_results))
