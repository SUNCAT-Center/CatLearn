"""Dimer: Diffusion along rows"""
from __future__ import print_function
import numpy as np

from math import sqrt

from ase import Atoms
from ase.io import Trajectory, read
from ase.optimize import LBFGS
from ase.dimer import DimerControl, MinModeAtoms, MinModeTranslate
from catlearn.optimize.functions_calc import Himmelblau
import matplotlib.pyplot as plt
from catlearn.optimize.mldimer import MLDimer


def get_plot(filename):

    """ Function for plotting each step of the toy model Muller-Brown .
    """

    fig = plt.figure(figsize=(10, 16))
    ax1 = plt.subplot()

    # Grid test points (x,y).
    x_lim = [-6., 6.]
    y_lim = [-6., 6.]

    # Axes.
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    ax1.set_xlim(x_lim)
    ax1.set_ylim(y_lim)

    # Range of energies:
    min_color = -0.1
    max_color = +15.

    # Length of the grid test points (nxn).
    test_points = 50

    test = []
    testx = np.linspace(x_lim[0], x_lim[1], test_points)
    testy = np.linspace(y_lim[0], y_lim[1], test_points)
    for i in range(len(testx)):
        for j in range(len(testy)):
            test1 = [testx[i], testy[j], 0.0]
            test.append(test1)
    test = np.array(test)

    x = []
    for i in range(len(test)):
        t = test[i][0]
        x.append(t)
    y = []
    for i in range(len(test)):
        t = test[i][1]
        y.append(t)

    # Plot real function.

    energy_ase = []
    for i in test:
        test_structure = Atoms('C', positions=[(i[0], i[1], i[2])])
        test_structure.set_calculator(Himmelblau())
        energy_ase.append(test_structure.get_potential_energy())

    crange = np.linspace(min_color, max_color, 1000)

    zi = plt.mlab.griddata(x, y, energy_ase, testx, testy, interp='linear')

    image = ax1.contourf(testx, testy, zi, crange, alpha=1., cmap='Spectral_r',
                         extend='neither', antialiased=False)
    for c in image.collections:
        c.set_edgecolor("face")
        c.set_linewidth(0.000001)

    crange2 = np.linspace(min_color, max_color, 10)
    ax1.contour(testx, testy, zi, crange2, alpha=0.6, antialiased=True)

    interval_colorbar = np.linspace(min_color, max_color, 5)

    fig.colorbar(image, ax=ax1, ticks=interval_colorbar, extend='None',
                 panchor=(0.5, 0.0),
                 orientation='horizontal', label='Function value (a.u.)')

    # Plot each point evaluated.
    all_structures = read(filename, ':')

    geometry_data = []
    for j in all_structures:
        geometry_data.append(j.get_positions().flatten())
    geometry_data = np.reshape(geometry_data, ((len(geometry_data)), 3))

    ax1.scatter(geometry_data[:, 0], geometry_data[:, 1], marker='x',
                s=50.0, c='black', alpha=1.0)

    plt.tight_layout(h_pad=1)
    plt.show()

##############################################################

# Setting up the initial image:
a = 4.0614
b = a / sqrt(2)
h = b / 2
atoms = Atoms('C', positions=[(-4., 4., 0)])

N = len(atoms)  # number of atoms

# Calculate using EMT:
atoms.set_calculator(Himmelblau())

# Relax the initial state:
LBFGS(atoms, trajectory='relaxation.traj').run(fmax=0.05)


# Settings Dimer:

# Mask
d_mask = [True] * N

# Manual vector:
displacement_vector = np.zeros((N, 3))
# Strength of displacement along y axis = along row:
displacement_vector[-1, 0] = -1.0
displacement_vector[-1, 1] = -1.0

# Random vector:
# step_displ = 1.0
# displacement_vector = np.random.uniform(low=-step_displ,
#                                         high=step_displ,
#                                         size=(N, 3))
# displacement_vector[0][2] = 0.0

# Dimer method using CatLearn.

ml_dimer = MLDimer(x0=atoms, ase_calc=Himmelblau(),
                   trajectory='mldimer_opt.traj')
ml_dimer.run(fmax=0.01, dmask=d_mask, vector=displacement_vector)

# Dimer method using ASE.

atoms.set_calculator(Himmelblau())

# Set trajectory
traj = Trajectory('dimer_optimization.traj', 'w', atoms)
traj.write()

# Set up the dimer:
d_control = DimerControl(initial_eigenmode_method='displacement',
                         displacement_method='vector',
                         logfile=None,
                         mask=d_mask)
d_atoms = MinModeAtoms(atoms, d_control)

d_atoms.displace(displacement_vector=displacement_vector)

# Converge to a saddle point:
dim_rlx = MinModeTranslate(d_atoms,
                           trajectory=traj,
                           logfile='-')
dim_rlx.run(fmax=0.01, steps=1000)

get_plot('dimer_optimization.traj')


iter_catlearn = len(read('mldimer_opt.traj', ':'))
iter_ase = len(read('dimer_optimization.traj', ':'))


print('Number of iterations performed by ASE Dimer:', iter_ase)
print('Number of iterations performed by CatLearn ML-Dimer:', ml_dimer.iter+1)

