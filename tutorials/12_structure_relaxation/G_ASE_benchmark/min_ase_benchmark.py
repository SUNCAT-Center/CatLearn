from ase import Atoms
from gpaw import GPAW
from ase.optimize import *
from ase.optimize.sciopt import *
from catlearn.optimize.catlearn_minimizer import CatLearnMinimizer
from ase.io import read
import copy
from ase.db import connect
import ase.db
from ase.visualize import view
import numpy as np
import os
import shutil

""" 
   ASE Benchmark.
"""

catlearn_version = '_u_1_2_0'

# Systems = ['H2', 'Au8CO', 'Cu2']
system = '_C5H12'

# 1. Setup calculator:
###############################################################################
calc = GPAW(mode='lcao',
     basis='dzp',
     kpts={'density': 2.0})

# 2. Build/Load structure:
###############################################################################
db = ase.db.connect('systems.db')
mol = db.get_atoms(formula='C5H12')

np.random.seed(1)
for i in mol:
    if i.position[2] > 8.50:
        i.position = i.position + np.random.normal(scale=0.0)

# 3. Benchmark.
###############################################################################
results_dir = './Results/'

if not os.path.exists(results_dir):
    os.makedirs(results_dir)


minimizers = ['BFGS', 'FIRE', 'LBFGS']

for i in minimizers:
    filename = i + system + '_opt.traj'
    if not os.path.exists(results_dir + filename):
        initial = mol.copy()
        initial.set_calculator(calc)
        opt = eval(i)(initial, trajectory=filename, )
        opt.run(fmax=0.02, steps=500)
        shutil.copy('./' + filename, results_dir + filename)

minimizers = ['BFGS', 'FIRE']

for i in minimizers:
    filename = i + system + '_opt' + catlearn_version
    if not os.path.exists(results_dir + filename + '_catlearn.traj'):
        initial = mol.copy()
        initial.set_calculator(calc)
        opt = CatLearnMinimizer(initial, filename=filename)
        opt.run(fmax=0.02, ml_algo=i)
        shutil.copy('./' + filename + '_catlearn.traj',
                    results_dir + filename + '_catlearn.traj')


# 4. Summary of the results:
###############################################################################

print('\n Summary of the results:\n ------------------------------------')

# Function evaluations:
for filename in os.listdir(results_dir):
    atoms = read(filename,':')
    feval = len(atoms)
    print('Number of function evaluations using ' + filename + ':', feval)