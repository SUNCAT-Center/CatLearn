from ase import Atoms
from gpaw import GPAW
from ase.optimize import *
from ase.optimize.sciopt import *
from catlearn.optimize.catlearn_minimizer import CatLearnMin
from catlearn.optimize.gptools_optimizer import GPTMin

from ase.io import read
import copy
import shutil
import os
from ase.constraints import FixAtoms
from ase.visualize import view
import ase.db


""" 
    Benchmark GPAW H2O calculations.
"""

calculator = GPAW(mode='lcao',
                  basis='dzp',
                  kpts={'density': 2.0})


# 1. Structural relaxation. ##################################

# 1.1. Set up structure:
a = 6
b = a / 2
initial_structure = Atoms('H2O',
            [(b, 0.7633 + b, -0.4876 + b),
             (b, -0.7633 + b, -0.4876 + b),
             (b, b, 0.1219 + b)],
                    cell=[a, a, a])
initial_structure.rattle(stdev=0.1, seed=3)
##############################################################################

# 2.A. Optimize structure using CatLearn:
initial_catlearn = initial_structure.copy()
initial_catlearn.set_calculator(calculator)

catlearn_opt = GPTMin(initial_catlearn, trajectory='catlearn_opt.traj')
catlearn_opt.run(fmax=0.05)

# 2.B. Optimize structure using ASE.
initial_ase = initial_structure.copy()
initial_ase.set_calculator(calculator)

ase_opt = GPMin(initial_ase, trajectory='ase_opt.traj',
                update_hyperparams=True)
ase_opt.run(fmax=0.05)

# 3. Summary of the results:
print('\n Summary of the results:\n ------------------------------------')
print('Number of function evaluations using CatLearn:', catlearn_opt.feval)
print('Number of function evaluations using ASE:', ase_opt.function_calls)