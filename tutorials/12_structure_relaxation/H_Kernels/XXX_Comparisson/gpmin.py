from ase import Atoms
from gpaw import GPAW
from ase.optimize import BFGS
from ase.optimize.sciopt import *
from ase.io import read, write
import copy
from catlearn.optimize.catlearn_minimizer import CatLearnMinimizer
from ase.optimize.gpmin.gpmin import GPMin


""" 
    Toy model minimization of a GPAW calculation.
"""

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

mol.rattle(seed=0, stdev=0.0)

# 2.A. Optimize structure using ASE.
initial_catlearn = mol.copy()
initial_catlearn.set_calculator(calc)
ase_opt = CatLearnMinimizer(initial_catlearn,
                            trajectory='results_catlearn.traj',
                            ml_calc='SQE_sequential',)
ase_opt.run(fmax=0.01, ml_algo='FIRE')

# 2.B Optimize using GPMin.
initial_gpmin = mol.copy()
initial_gpmin.set_calculator(calc)
gpmin_opt = GPMin(initial_gpmin, trajectory='results_gpmin.traj')
gpmin_opt.run(fmax=0.01)

# 3. Summary of the results:
print('\n Summary of the results:\n ------------------------------------')

gpmin_results = read('results_gpmin.traj', ':')

print('Number of function evaluations using GPMin:', len(gpmin_results))
print('Energy GPMin (eV):', gpmin_results[-1].get_potential_energy())

catlearn_results = read('results_catlearn.traj', ':')

print('Number of function evaluations using CatLearn:', len(catlearn_results))
print('Energy CatLearn (eV):', catlearn_results[-1].get_potential_energy())
