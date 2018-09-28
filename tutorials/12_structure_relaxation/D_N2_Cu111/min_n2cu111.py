from catlearn.optimize.catlearn_minimizer import CatLearnMin
from ase.io import read
from ase.optimize import *
from ase.optimize.sciopt import *
from ase.visualize import view
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

# 1. Structural relaxation. ##################################################

# Setup calculator.
calculator = EMT()

# 1.1. Structures:

h = 1.85
d = 1.10

slab = fcc111('Cu', size=(2, 2, 4), vacuum=10.0)
molecule = Atoms('CO', positions=[(0., 0., 0.), (0., 0., d)])
molecule.rattle(stdev=0.1, seed=1)
add_adsorbate(slab, molecule, h, 'ontop')

constraint = FixAtoms(mask=[atom.position[2] < 14.0 for atom in slab])
slab.set_constraint(constraint)

initial_structure = slab.copy()

# 2.A. Optimize structure using CatLearn:

initial_catlearn = initial_structure.copy()
initial_catlearn.set_calculator(calculator)

catlearn_opt = CatLearnMin(initial_catlearn, trajectory='catlearn_opt.traj')
catlearn_opt.run(fmax=0.05, kernel='SQE_opt')

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
print('Number of function evaluations using ASE:', ase_opt.function_calls)

