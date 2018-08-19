from catlearn.optimize.catlearn_minimizer import CatLearnMinimizer
from ase.calculators.emt import EMT
from ase.io import read
from ase.optimize import BFGS
from ase.visualize import view
import copy
from ase import Atoms
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.optimize import QuasiNewton
from ase.build import fcc111, add_adsorbate

""" 
    Toy model minimization of N2 on Cu111.
    Minimization example. 
"""

# 1. Structural relaxation. ##################################################

# Setup calculator:
ase_calculator = EMT()

# 1.1. Structures:

h = 1.85
d = 1.10

slab = fcc111('Cu', size=(2, 2, 4), vacuum=10.0)
molecule = Atoms('CO', positions=[(0., 0., 0.), (0., 0., d)])
molecule.rattle(stdev=0.0, seed=0)
add_adsorbate(slab, molecule, h, 'ontop')

constraint = FixAtoms(mask=[atom.position[2] < 14.0 for atom in slab])
slab.set_constraint(constraint)

common_initial = slab.copy()

# 2.A. Optimize structure using ASE.
initial_ase = copy.deepcopy(common_initial)
initial_ase.set_calculator(copy.deepcopy(ase_calculator))

ase_opt = BFGS(initial_ase, trajectory='ase_optimization.traj')
ase_opt.run(fmax=0.01)

# visual_ase = read('ase_optimization.traj', ':')
# view(visual_ase)
# exit()

# 2.B. Optimize structure using CatLearn:

initial_catlearn = copy.deepcopy(common_initial)
initial_catlearn.set_calculator(copy.deepcopy(ase_calculator))

catlearn_opt = CatLearnMinimizer(initial_catlearn, filename='results')
catlearn_opt.run(fmax=0.01)

# 3. Summary of the results:
print('\n Summary of the results:\n ------------------------------------')

ase_results = read('ase_optimization.traj', ':')

print('Number of function evaluations using ASE:', len(ase_results))
print('Energy ASE (eV):', ase_results[-1].get_potential_energy())

catlearn_results = read('results_catlearn.traj', ':')

print('Number of function evaluations using CatLearn:', len(catlearn_results))
print('Energy CatLearn (eV):', catlearn_results[-1].get_potential_energy())
