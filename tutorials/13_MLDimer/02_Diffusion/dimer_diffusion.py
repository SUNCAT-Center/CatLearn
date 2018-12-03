"""Dimer: Diffusion along rows"""
from __future__ import print_function
import numpy as np

from math import sqrt

from ase import Atoms, Atom
from ase.io import Trajectory, read, write
from ase.constraints import FixAtoms
from ase.optimize import QuasiNewton
from ase.calculators.emt import EMT
from ase.dimer import DimerControl, MinModeAtoms, MinModeTranslate
from catlearn.optimize.mldimer import MLDimer

# Setting up the initial image:
a = 4.0614
b = a / sqrt(2)
h = b / 2
initial = Atoms('Al2',
                positions=[(0, 0, 0),
                           (a / 2, b / 2, -h)],
                cell=(a, b, 2 * h),
                pbc=(1, 1, 0))
initial *= (2, 2, 2)
initial.append(Atom('Al', (a / 2, b / 2, 3 * h)))
initial.center(vacuum=4.0, axis=2)

N = len(initial)  # number of atoms

# Make a mask of zeros and ones that select fixed atoms - the two
# bottom layers:
mask = initial.positions[:, 2] - min(initial.positions[:, 2]) < 1.5 * h
constraint = FixAtoms(mask=mask)
initial.set_constraint(constraint)

# Calculate using EMT:
initial.set_calculator(EMT())

# Relax the initial state:
QuasiNewton(initial, trajectory='initial.traj').run(fmax=0.05)
e0 = initial.get_potential_energy()

traj = Trajectory('dimer_along.traj', 'w', initial)
traj.write()

# Making dimer mask list:
d_mask = [False] * (N - 1) + [True]

# Set up the dimer:
d_control = DimerControl(initial_eigenmode_method='displacement',
                         displacement_method='vector',
                         logfile=None,
                         mask=d_mask)
d_atoms = MinModeAtoms(initial, d_control)

# Displacement settings:
displacement_vector = np.zeros((N, 3))
# Strength of displacement along y axis = along row:
displacement_vector[-1, 0] = 0.001
# The direction of the displacement is set by the a in
# displacement_vector[-1, a], where a can be 0 for x, 1 for y and 2 for z.
d_atoms.displace(displacement_vector=displacement_vector)

# Converge to a saddle point:
dim_rlx = MinModeTranslate(d_atoms,
                           trajectory=traj,
                           logfile='-')
dim_rlx.run(fmax=0.001)

diff = initial.get_potential_energy() - e0
print(('The energy barrier is %f eV.' % diff))

# CatLearn MLDimer:
atoms = read('initial.traj')
ml_dimer = MLDimer(x0=atoms, ase_calc=EMT(),
                   trajectory='mldimer_opt.traj')
ml_dimer.run(fmax=0.001, dmask=d_mask, vector=displacement_vector)

# Summary:
iter_catlearn = len(read('mldimer_opt.traj', ':'))
iter_ase = len(read('dimer_along.traj', ':'))


print('Number of iterations performed by ASE Dimer:', iter_ase)
print('Number of iterations performed by CatLearn ML-Dimer:', iter_catlearn)



