from ase.build import fcc111, add_adsorbate
from ase.visualize import view
from ase import Atoms
from ase.constraints import FixAtoms
from ase.constraints import FixBondLength
from ase.calculators.emt import EMT
from ase.io import read
from ase.optimize import *
from catlearn.optimize.mlmin import MLMin

""" 
    Structure relaxation using constraints. CO on an Al(111) surface.
    Fixed bond length constraint.
    Benchmark using MLMin, GPMin, LBFGS and FIRE. 
"""


# 1. Build Atoms Object.
###############################################################################

# Setup calculator:
calc = EMT()

# Create CO molecule:
d = 1.1
co = Atoms('CO', positions=[(0, 0, 0), (0, 0, d)])

# Create slab:
slab = fcc111('Al', size=(2,2,3), vacuum=10.0)
slab = fcc111('Al', size=(2,2,3))

# Add CO on the slab:
add_adsorbate(slab, co, 2., 'bridge')
slab.center(vacuum=10.0, axis=2)

# Set constraints:
c1 = FixAtoms(indices=[atom.index for atom in slab if atom.symbol == 'Al'])
c2 = FixBondLength(12, 13)
slab.set_constraint([c1, c2])

atoms = slab.copy()

# 2. Benchmark.
###############################################################################

# 2.A. Optimize structure using MLMin (CatLearn).
initial_mlmin = atoms.copy()
initial_mlmin.set_calculator(calc)
mlmin_opt = MLMin(initial_mlmin, trajectory='results_catlearn.traj')
mlmin_opt.run(fmax=0.01, kernel='SQE')

final_atoms = read('results_catlearn.traj', ':')

# 2.B Optimize using GPMin.
initial_gpmin = atoms.copy()
initial_gpmin.set_calculator(calc)
gpmin_opt = GPMin(initial_gpmin, trajectory='results_gpmin.traj',
                  update_hyperparams=True)
gpmin_opt.run(fmax=0.01)

# 2.C Optimize using LBFGS.
initial_lbfgs = atoms.copy()
initial_lbfgs.set_calculator(calc)
lbfgs_opt = BFGS(initial_lbfgs, trajectory='results_lbfgs.traj')
lbfgs_opt.run(fmax=0.01)

# 2.D Optimize using FIRE.
initial_fire = atoms.copy()
initial_fire.set_calculator(calc)
fire_opt = FIRE(initial_fire, trajectory='results_fire.traj')
fire_opt.run(fmax=0.01)

# 3. Summary of the results:
###############################################################################

print('\n Summary of the results:\n ------------------------------------')

fire_results = read('results_fire.traj', ':')
print('Number of function evaluations using FIRE:',
      len(fire_results))

lbfgs_results = read('results_lbfgs.traj', ':')
print('Number of function evaluations using LBFGS:',
      len(lbfgs_results))

gpmin_results = read('results_gpmin.traj', ':')
print('Number of function evaluations using GPMin:', gpmin_opt.function_calls)

catlearn_results = read('results_catlearn.traj', ':')
print('Number of function evaluations using MLMin (CatLearn):',
      len(catlearn_results))






