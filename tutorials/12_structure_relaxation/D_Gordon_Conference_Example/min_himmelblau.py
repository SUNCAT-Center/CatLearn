from catlearn.optimize.catlearn_minimizer import MLOptimizer
from catlearn.optimize.functions_calc import ModifiedHimmelblau, Rosenbrock
from ase import Atoms
from ase.optimize import BFGS, FIRE, LBFGS, MDMin
from ase.io import read
import copy
import numpy as np
import matplotlib.pyplot as plt
from catlearn.optimize.io import array_to_ase
from catlearn.optimize.catlearn_ase_calc import CatLearnASE

""" 
    Minimization example.
    Toy model using toy model potentials such as Muller-Brown, Himmelblau, 
    Goldstein-Price or Rosenbrock.       
"""

# 0. Set calculator.
ase_calculator = ModifiedHimmelblau()

# 1. Set common initial structure.
common_initial = Atoms('C', positions=[(-1.5, -1.0, 0.0)])

# 2.A. Optimize structure using ASE.
initial_ase = copy.deepcopy(common_initial)
initial_ase.set_calculator(copy.deepcopy(ase_calculator))

ase_opt = MDMin(initial_ase, trajectory='ase_optimization.traj')
ase_opt.run(fmax=0.01)

# # Plot each step in a folder.
ase_trajectory = read('ase_optimization.traj', ':')

opt_ase_positions = []
opt_ase_energies = []
for i in range(0, len(ase_trajectory)):
    opt_ase_positions.append(ase_trajectory[i].get_positions()[0][0:2])
    opt_ase_energies.append(ase_trajectory[i].get_potential_energy())

opt_ase_positions = np.reshape(opt_ase_positions, (len(opt_ase_positions), 2))

folder = './plots_ase/'

plt.figure(figsize=(4.0, 4.0))

# Contour plot for real function.
limitsx = [-5.5, 5.5]
limitsy = [-5.5, 5.5]
crange = np.linspace(-0.5, 6.0, 60)
crange2 = 40

plot_resolution = 150
A = np.linspace(limitsx[0], limitsx[1], plot_resolution)
B = np.linspace(limitsy[0], limitsy[1], plot_resolution)
X, Y = np.meshgrid(A, B)
Z = np.zeros((len(A),len(B)))
for i in range(0, len(A)):
    for j in range(0,len(B)):
        struc_i = Atoms('C', positions=[(A[j], B[i], 0.0)])
        struc_i.set_calculator(copy.deepcopy(ase_calculator))
        e = struc_i.get_total_energy()
        Z[i][j] = e
plt.contourf(X, Y, Z, crange, alpha=1.0, cmap='terrain')
plt.contour(X, Y, Z, crange2, alpha=0.5, linewidths=1.2, antialiased=True,
            linestyles='dashed')

plt.xlim(limitsx[0], limitsx[1])
plt.ylim(limitsy[0], limitsy[1])
plt.xticks([])
plt.yticks([])
plt.savefig(fname=(folder+'widerange_himmelblau.png'), dpi=500, format='png',
            transparent=False)
plt.show()



def plot_ase_steps(iteration, positions):
    plt.figure(figsize=(4.0, 4.0))
    limitsx = [-5.5, 0]
    limitsy = [-5.5, 0]
    crange = np.linspace(-0.5, 10.0, 60)
    crange2 = 40

    plot_resolution = 150
    A = np.linspace(limitsx[0], limitsx[1], plot_resolution)
    B = np.linspace(limitsy[0], limitsy[1], plot_resolution)
    X, Y = np.meshgrid(A, B)
    Z = np.zeros((len(A),len(B)))
    for i in range(0, len(A)):
        for j in range(0,len(B)):
            x = [A[j], B[i]]
            e = 0.05 * ((x[0]**2 + x[1] -11)**2 + (x[0] + x[1]**2 -7)**2 \
                 + 0.5 * x[0] + x[1])
            Z[i][j] = e
    plt.contourf(X, Y, Z, crange, alpha=1.0, cmap='terrain')
    plt.contour(X, Y, Z, crange2, alpha=0.5, linewidths=1.0, antialiased=True,
                linestyles='dashed')
    plt.scatter(positions[-1, 0], positions[-1, 1],
                marker='o', c='white', edgecolors='black', alpha=1.0)
    plt.plot(positions[:, 0], positions[:, 1], linewidth=2.0, color='black')
    plt.xlim(limitsx[0], limitsx[1])
    plt.ylim(limitsy[0], limitsy[1])
    plt.xticks([])
    plt.yticks([])
    plt.savefig(fname=(folder+'min_ase_himmelblau_iter'+ str(iteration) +
                '.png'), dpi=500, format='png', transparent=False)
    plt.show()


for i in range(0, len(opt_ase_positions)):
    opt_ase_positions_i = opt_ase_positions[0:i+1]
    plot_ase_steps(iteration=i, positions=opt_ase_positions_i)

# 2.B. Optimize structure using CatLearn:

initial_catlearn = copy.deepcopy(common_initial)
initial_catlearn.set_calculator(copy.deepcopy(ase_calculator))
catlearn_opt = MLOptimizer(initial_catlearn, filename='results')
catlearn_opt.run(fmax=0.01, ml_algo='MDMin')

# 3. Summary of the results:
print('\n Summary of the results:\n ------------------------------------')

ase_results = read('ase_optimization.traj', ':')

print('Number of function evaluations using ASE:', len(ase_results))
print('Energy ASE (eV):', ase_results[-1].get_potential_energy())
print('Coordinates ASE:', ase_results[-1].get_positions().flatten())

catlearn_results = read('results_catlearn.traj', ':')

print('Number of function evaluations using CatLearn:', len(catlearn_results))
print('Energy CatLearn (eV):', catlearn_results[-1].get_potential_energy())
print('Coordinates CatLearn:', catlearn_results[-1].get_positions().flatten())