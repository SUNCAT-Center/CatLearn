from catlearn.optimize.catlearn_minimizer import CatLearnMinimizer
from ase.calculators.emt import EMT
from ase.io import read
from ase.optimize import *
from ase.optimize.sciopt import *
from ase.visualize import view
import copy
from ase import Atoms
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.build import fcc111, add_adsorbate
import os
import shutil

""" 
    Toy model minimization of N2 on Cu111.
    Minimization example. 
"""

catlearn_version = '_u_1_0_9'

results_dir = './Results/'

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# 1. Structural relaxation. ##################################################

# Setup calculator.
calc = EMT()

# 1.1. Structures:

h = 1.85
d = 1.10

slab = fcc111('Cu', size=(2, 2, 4), vacuum=10.0)
molecule = Atoms('CO', positions=[(0., 0., 0.), (0., 0., d)])
molecule.rattle(stdev=0.4, seed=0)
add_adsorbate(slab, molecule, h, 'ontop')

constraint = FixAtoms(mask=[atom.position[2] < 14.0 for atom in slab])
slab.set_constraint(constraint)

mol = slab.copy()


## 2. Benchmark.

minimizers = ['BFGS', 'LBFGS', 'SciPyFminCG', 'SciPyFminBFGS',
              'FIRE', 'QuasiNewton', 'BFGSLineSearch', 'GoodOldQuasiNewton']

for i in minimizers:
    filename = i + '_opt.traj'
    if not os.path.exists(results_dir + filename):
        initial = mol.copy()
        initial.set_calculator(calc)
        opt = eval(i)(initial, trajectory=filename)
        opt.run(fmax=0.05, steps=500)
        shutil.copy('./' + filename, results_dir + filename)


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
