from catlearn.optimize.catlearn_minimizer import CatLearnMin
from ase.calculators.emt import EMT
from ase.io import read
from ase.optimize import BFGS, FIRE, MDMin
from ase.optimize.sciopt import SciPyFminPowell, SciPyFminBFGS
from ase.visualize import view
import copy
from ase.optimize.gpmin.gpmin import GPMin


""" 
    Toy model minimization of Pt 19 cluster.
    Minimization example. 
"""

# 1. Structural relaxation. ##################################################

# Setup calculator:
calc = EMT()

# 1.1. Structures:

mol = read('./A_structure/POSCAR')
mol.rattle(stdev=0.1, seed=30)

# 3. Benchmark.
###############################################################################

# 2.A. Optimize structure using ASE.

initial_catlearn = mol.copy()
initial_catlearn.set_calculator(calc)
ase_opt = CatLearnMin(initial_catlearn,
                            trajectory='results_catlearn.traj',
                            ml_calc='SQE_sequential')
ase_opt.run(fmax=0.01)
atoms = read('results_catlearn.traj', ':')



# 2.B Optimize using GPMin.
initial_gpmin = mol.copy()
initial_gpmin.set_calculator(calc)
gpmin_opt = GPMin(initial_gpmin, trajectory='results_gpmin.traj',
                  update_hyperparams=False)
gpmin_opt.run(fmax=0.01)


# 3. Summary of the results:
print('\n Summary of the results:\n ------------------------------------')


catlearn_results = read('results_catlearn.traj', ':')

print('Number of function evaluations using CatLearn:', len(catlearn_results))
print('Energy CatLearn (eV):', catlearn_results[-1].get_potential_energy())

gpmin_results = read('results_gpmin.traj', ':')

print('Number of function evaluations using GPMin:', len(gpmin_results))
print('Energy GPMin (eV):', gpmin_results[-1].get_potential_energy())