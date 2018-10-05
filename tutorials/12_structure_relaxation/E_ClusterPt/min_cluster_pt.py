from catlearn.optimize.catlearn_minimizer import CatLearnMin
from ase.calculators.emt import EMT
from ase.io import read
from ase.optimize import BFGS, FIRE, MDMin
from ase.optimize.sciopt import SciPyFminPowell, SciPyFminBFGS, SciPyFminCG
from ase.visualize import view
import copy
from ase.optimize.gpmin.gpmin import GPMin


""" 
    Toy model minimization of Pt 19 cluster.
    Minimization example. 
"""

# Setup calculator:
calc = EMT()

# 1.1. Structures:

mol = read('./A_structure/POSCAR')
mol.rattle(stdev=0.1, seed=0)

# 2.A. Optimize structure using ASE.
initial_catlearn = mol.copy()
initial_catlearn.set_calculator(calc)
catlearn_opt = CatLearnMin(initial_catlearn, trajectory='catlearn_opt.traj')
catlearn_opt.run(fmax=0.05)

# 2.B Optimize using ASE.
initial_ase = mol.copy()
initial_ase.set_calculator(calc)
ase_opt = GPMin(initial_ase, trajectory='ase_opt.traj',
                update_hyperparams=False)
ase_opt.run(fmax=0.05)

# 3. Summary of the results:
print('\n Summary of the results:\n ------------------------------------')

catlearn_results = read('catlearn_opt.traj', ':')

print('Number of function evaluations using CatLearn:', catlearn_opt.feval)
print('Energy CatLearn (eV):', catlearn_results[-1].get_potential_energy())

ase_results = read('ase_opt.traj', ':')

print('Number of function evaluations using ASE:', ase_opt.function_calls)
print('Energy ASE (eV):', ase_results[-1].get_potential_energy())
