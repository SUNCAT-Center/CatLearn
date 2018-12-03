from ase.build import fcc111
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.optimize import *
from ase.build import add_adsorbate
from ase import Atoms, Atom
from ase.io import Trajectory, read, write
from ase.constraints import FixAtoms
from ase.optimize import QuasiNewton
from ase.calculators.emt import EMT
from ase.dimer import DimerControl, MinModeAtoms, MinModeTranslate
from catlearn.optimize.mldimer import MLDimer
import numpy as np
from ase.visualize import view

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

# Calculate using EMT:
atoms.set_calculator(EMT())

# Relax the initial state:
QuasiNewton(atoms, trajectory='initial.traj').run(fmax=0.05)
e0 = atoms.get_potential_energy()

# Dimer calculations:

N = len(atoms)  # number of atoms

traj = Trajectory('dimer_along.traj', 'w', atoms)
traj.write()

# Making dimer mask list:
d_mask = [False] * (N - 1) + [True]

# Set up the dimer:
d_control = DimerControl(initial_eigenmode_method='displacement',
                         displacement_method='vector',
                         logfile=None,
                         mask=d_mask)
d_atoms = MinModeAtoms(atoms, d_control)

# Displacement settings:
displacement_vector = np.zeros((N, 3))
# Strength of displacement along y axis = along row:
displacement_vector[-1, 1] = 0.1
displacement_vector[-1, 0] = 0.5

# The direction of the displacement is set by the a in
# displacement_vector[-1, a], where a can be 0 for x, 1 for y and 2 for z.
d_atoms.displace(displacement_vector=displacement_vector)

# Converge to a saddle point:
dim_rlx = MinModeTranslate(d_atoms,
                           trajectory=traj,
                           logfile='-')
# dim_rlx.run(fmax=0.01)

# diff = atoms.get_potential_energy() - e0
# print(('The energy barrier is %f eV.' % diff))

# CatLearn MLDimer:
atoms = read('initial.traj')
ml_dimer = MLDimer(x0=atoms, ase_calc=EMT(),
                   trajectory='mldimer_opt.traj')
ml_dimer.run(fmax=0.01, dmask=d_mask, vector=displacement_vector)

# Summary:
final_catlearn = read('mldimer_opt.traj', ':')
iter_catlearn = len(final_catlearn)
final_ase = read('dimer_along.traj', ':')
iter_ase = len(final_ase)

print('Number of iterations performed by ASE Dimer:', iter_ase)
print('Number of iterations performed by CatLearn ML-Dimer:', iter_catlearn)

view(final_ase)
view(final_catlearn)



