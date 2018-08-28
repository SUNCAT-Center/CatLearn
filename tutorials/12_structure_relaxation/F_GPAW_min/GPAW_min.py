from ase import Atoms
from gpaw import GPAW
from ase.optimize import *
from ase.optimize.sciopt import *
from catlearn.optimize.catlearn_minimizer import CatLearnMinimizer
from ase.io import read
import copy
import shutil
import os
from ase.constraints import FixAtoms
from ase.visualize import view

""" 
    Benchmark GPAW H2O calculations.
"""

catlearn_version = '_u_1_8_0_sequential'

results_dir = './Results/'

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# 0.0 Setup calculator:
calc = GPAW(nbands=4, h=0.2, mode='lcao', basis='dzp')

# 1. Structural relaxation. ##################################################

# 1.1. Set up structure:
a = 6
b = a / 2
mol = Atoms('H2O',
            [(b, 0.7633 + b, -0.4876 + b),
             (b, -0.7633 + b, -0.4876 + b),
             (b, b, 0.1219 + b)],
            cell=[a, a, a])

mol.rattle(seed=0, stdev=0.2)

##############################################################################

minimizers = ['BFGS', 'LBFGS', 'FIRE']

for i in minimizers:
    filename = i + '_opt.traj'
    if not os.path.exists(results_dir + filename):
        initial = mol.copy()
        initial.set_calculator(calc)
        opt = eval(i)(initial, trajectory=filename)
        opt.run(fmax=0.05)
        shutil.copy('./' + filename, results_dir + filename)

minimizers = ['FIRE', 'BFGS', 'LBFGS']

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
