from ase import Atoms
from gpaw import GPAW
from ase.optimize import BFGS
from ase.optimize.sciopt import *
from catlearn.optimize.catlearn_minimizer import CatLearnMinimizer
from ase.io import read, write
import copy

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

mol.rattle(seed=0, stdev=0.1)


# 2.A. Optimize structure using ASE.
initial_ase = mol.copy()
initial_ase.set_calculator(calc)

ase_opt = SciPyFminCG(initial_ase, trajectory='ase_optimization.traj')
ase_opt.run(fmax=0.05)

# 2.B. Optimize structure using CatLearn:
initial_catlearn = mol.copy()
initial_catlearn.set_calculator(calc)

catlearn_opt = CatLearnMinimizer(initial_catlearn,
                                 filename='results')
catlearn_opt.run(fmax=0.05)

# 3. Summary of the results:
print('\n Summary of the results:\n ------------------------------------')

ase_results = read('ase_optimization.traj', ':')

print('Number of function evaluations using ASE:', len(ase_results))
print('Energy ASE (eV):', ase_results[-1].get_potential_energy())

catlearn_results = read('results_catlearn.traj', ':')

print('Number of function evaluations using CatLearn:', len(catlearn_results))
print('Energy CatLearn (eV):', catlearn_results[-1].get_potential_energy())
