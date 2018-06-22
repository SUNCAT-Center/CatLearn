from catlearn.optimize.catlearn_minimizer import MLOptimizer
from catlearn.optimize.functions_calc import ModifiedHimmelblau, NoiseHimmelblau
from ase import Atoms
from ase.optimize import BFGS, FIRE, LBFGS, MDMin
from ase.io import read
import copy
import numpy as np
import matplotlib.pyplot as plt
from catlearn.optimize.io import array_to_ase
from catlearn.optimize.catlearn_ase_calc import CatLearnASE
from catlearn.optimize.convergence import get_fmax

""" 
    Minimization example.
    Toy model using toy model potentials such as Muller-Brown, Himmelblau, 
    Goldstein-Price or Rosenbrock.       
"""

# 0. Set calculator.
ase_calculator = NoiseHimmelblau()

# 1. Set common initial structure.
common_initial = Atoms('C', positions=[(-1.5, -1.0, 0.0)])

# 2.A. Optimize structure using ASE.
initial_ase = copy.deepcopy(common_initial)
initial_ase.set_calculator(copy.deepcopy(ase_calculator))

ase_opt = FIRE(initial_ase, trajectory='ase_optimization.traj')
ase_opt.run(fmax=0.01)

# # Plot each step in a folder.
ase_trajectory = read('ase_optimization.traj', ':')

opt_ase_positions = []
opt_ase_energies = []
opt_ase_forces = []
for i in range(0, len(ase_trajectory)):
    opt_ase_positions.append(ase_trajectory[i].get_positions()[0][0:2])
    opt_ase_energies.append(ase_trajectory[i].get_potential_energy())
    opt_ase_forces.append(ase_trajectory[i].get_forces())

opt_ase_positions = np.reshape(opt_ase_positions, (len(opt_ase_positions), 2))
opt_ase_forces = np.reshape(opt_ase_forces, (len(opt_ase_forces), 3))
list_fmax_ase = get_fmax(opt_ase_forces, 1)

folder = './plots_ase/'

plt.figure(figsize=(4.0, 4.0))

# Contour plot for real function.
limitsx = [-5.5, 5.5]
limitsy = [-5.5, 5.5]
crange = np.linspace(-0.5, 10.0, 60)
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
plt.close()


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
            struc_i = Atoms('C', positions=[(A[j], B[i], 0.0)])
            struc_i.set_calculator(copy.deepcopy(ase_calculator))
            e = struc_i.get_total_energy()
            Z[i][j] = e
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
    plt.close()

for i in range(0, len(opt_ase_positions)):
    opt_ase_positions_i = opt_ase_positions[0:i+1]
    plot_ase_steps(iteration=i, positions=opt_ase_positions_i)

# # # 2.B. Optimize structure using CatLearn:
# #
initial_catlearn = copy.deepcopy(common_initial)
initial_catlearn.set_calculator(copy.deepcopy(ase_calculator))
catlearn_opt = MLOptimizer(initial_catlearn, filename='results')
catlearn_opt.run(fmax=0.01, ml_algo='FIRE')

energies_catlearn = catlearn_opt.list_targets
forces_catlearn = catlearn_opt.list_gradients
list_fmax_catlearn = get_fmax(forces_catlearn, 1)


# Plot of the energy convergence:
list_iterations = np.arange(0, len(opt_ase_energies))


for i in range(0, len(opt_ase_energies)):
    plt.figure(figsize=(4.0, 4.0))
    plt.xlim([0, len(opt_ase_energies)+5])
    plt.ylim([-1.0, np.max(opt_ase_energies)+0.2])

    plt.plot(list_iterations[0:i+1], opt_ase_energies[0:i+1], c='black',
             lw=2.0)

    lt = list_iterations[0:i+1].copy()
    if len(list_iterations[0:i+1]) >= len(energies_catlearn):
        lt = list_iterations.copy()
        lt = lt[0:len(energies_catlearn)]

    plt.plot(lt, energies_catlearn[0:i+1], c='red',
             lw=2.0)
    plt.savefig(fname=(folder+'energies_himmelblau_iter'+ str(i) +
                '.png'), dpi=500, format='png', transparent=False)
    plt.xlim([0, len(opt_ase_energies)+5])
    plt.ylim([-1.0, np.max(opt_ase_energies)+0.2])
    plt.show()
    plt.close()


# Plot of the forces convergence:
for i in range(0, len(list_fmax_ase)):
    plt.xlim([0, len(list_fmax_ase)+5])
    plt.ylim([-1.0, np.max(list_fmax_ase)+0.2])
    plt.figure(figsize=(4.0, 4.0))
    plt.plot(list_iterations[0:i+1], list_fmax_ase[0:i+1], c='black',
             lw=2.0)

    lt = list_iterations[0:i+1].copy()
    if len(list_iterations[0:i+1]) >= len(list_fmax_catlearn):
        lt = list_iterations.copy()
        lt = lt[0:len(list_fmax_catlearn)]

    plt.plot(lt, list_fmax_catlearn[0:i+1], c='red',
             lw=2.0)
    plt.savefig(fname=(folder+'forces_himmelblau_iter'+ str(i) +
                '.png'), dpi=500, format='png', transparent=False)
    plt.xlim([0, len(list_fmax_ase)+5])
    plt.ylim([-1.0, np.max(list_fmax_ase)+0.2])
    plt.show()
    plt.close()


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