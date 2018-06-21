from catlearn.optimize.catlearn_minimizer import MLOptimizer
from ase.calculators.emt import EMT
from ase.io import read
from ase.optimize import BFGS, MDMin
import copy


""" 
    Toy model minimization of Pt heptamer island on Pt(111).
    Minimization example. 
"""

# 1. Structural relaxation. ##################################################

# Setup calculator:
ase_calculator = EMT()

# 1.1. Structures:

common_initial = read('./A_structure/POSCAR')
# common_initial.rattle(stdev=0.1, seed=0)

# 2.A. Optimize structure using ASE.
initial_ase = copy.deepcopy(common_initial)
initial_ase.set_calculator(copy.deepcopy(ase_calculator))

ase_opt = BFGS(initial_ase, trajectory='ase_optimization.traj')
ase_opt.run(fmax=0.01)

# 2.B. Optimize structure using CatLearn:

initial_catlearn = copy.deepcopy(common_initial)
initial_catlearn.set_calculator(copy.deepcopy(ase_calculator))

catlearn_opt = MLOptimizer(initial_catlearn, filename='results')
catlearn_opt.run(fmax=0.01, ml_algo='BFGS')

# 3. Summary of the results:
print('\n Summary of the results:\n ------------------------------------')

ase_results = read('ase_optimization.traj', ':')

print('Number of function evaluations using ASE:', len(ase_results))
print('Energy ASE (eV):', ase_results[-1].get_potential_energy())

catlearn_results = read('results_catlearn.traj', ':')

print('Number of function evaluations using CatLearn:', len(catlearn_results))
print('Energy CatLearn (eV):', catlearn_results[-1].get_potential_energy())
