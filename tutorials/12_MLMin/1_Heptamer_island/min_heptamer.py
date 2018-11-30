from ase.build import fcc111
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.io import read
from ase.optimize import *
from ase.build import add_adsorbate
from catlearn.optimize.mlmin import MLMin


""" 
    Structure relaxation of Pt heptamer island on Pt(111).
    Benchmark using MLMin, GPMin, LBFGS and FIRE. 
"""

# 1. Build Atoms Object.
###############################################################################

# Setup calculator:
calc = EMT()

# 1.1. Set up structure:

# Build a 3 layers 5x5-Pt(111) slab.
atoms = fcc111('Pt', size=(5, 5, 3))
c = FixAtoms(indices=[atom.index for atom in atoms if atom.symbol == 'Pt'])
atoms.set_constraint(c)

atoms.center(axis=2, vacuum=15.0)

# Build heptamer island:
atoms2 = fcc111('Au', size=(3, 3, 1))
atoms2.pop(0)
atoms2.pop(7)
atoms2.rattle(stdev=0.10, seed=0)

# Add island to slab:
add_adsorbate(atoms, atoms2, 2.5, offset=0.5)

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
lbfgs_opt = LBFGS(initial_lbfgs, trajectory='results_lbfgs.traj')
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


