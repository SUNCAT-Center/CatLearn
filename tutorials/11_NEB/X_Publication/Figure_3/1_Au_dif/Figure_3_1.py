import numpy as np
from ase.optimize import BFGS, MDMin, FIRE
from catlearn.optimize.catlearn_neb_optimizer import CatLearnNEB
from ase.calculators.emt import EMT
from ase.neb import NEB
from ase.constraints import FixAtoms
from ase.build import fcc100, add_adsorbate
import copy
from ase.io import read, write
import os
import shutil
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

""" 
    Figure 3.A. Number of function calls as a function of number of images. 
    Diffusion Au atom on Al(111).
"""

results_dir = './results/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# 1. Structural relaxation. ##################################################

# Setup calculator:
ase_calculator = EMT()

# 1.1. Structures:

# 2x2-Al(001) surface with 3 layers and an
# Au atom adsorbed in a hollow site:
slab = fcc100('Al', size=(2, 2, 3))
add_adsorbate(slab, 'Au', 1.7, 'hollow')
slab.center(axis=2, vacuum=4.0)
slab.set_calculator(copy.deepcopy(ase_calculator))

# Fix second and third layers:
mask = [atom.tag > 1 for atom in slab]
slab.set_constraint(FixAtoms(mask=mask))

# 1.2. Optimize initial and final end-points.

# Initial end-point:
qn = BFGS(slab, trajectory='initial_opt.traj')
qn.run(fmax=0.01)

# Final end-point:
slab[-1].x += slab.get_cell()[0, 0] / 2
qn = BFGS(slab, trajectory='final_opt.traj')
qn.run(fmax=0.01)

# Run NEBS.
n_images = [7, 11, 15, 19, 23, 27]

for n in n_images:

    # 2.A. NEB using ASE ######################################################

    for algo in ['MDMin', 'FIRE']:

        filename = 'neb_ase_' + algo + '_' + str(n) + '_images.traj'

        if not os.path.exists(results_dir + filename):
            initial_ase = read('initial_opt.traj')
            final_ase = read('final_opt.traj')
            constraint = FixAtoms(mask=[atom.tag > 1 for atom in initial_ase])

            images_ase = [initial_ase]
            for i in range(1, n-1):
                image = initial_ase.copy()
                image.set_calculator(copy.deepcopy(ase_calculator))
                image.set_constraint(constraint)
                images_ase.append(image)

            images_ase.append(final_ase)

            neb_ase = NEB(images_ase, climb=True)
            neb_ase.interpolate(method='idpp')

            qn_ase = eval(algo)(neb_ase, trajectory=filename)
            max_steps = 200
            qn_ase.run(fmax=0.05, steps=max_steps)

            # Save results of the run in traj file.
            atoms_ase = read(filename, ':')
            atoms_ase[-1].info['function_calls'] = (len(atoms_ase) - 2 * n)
            atoms_ase[-1].info['mean_error'] = 0.0
            atoms_ase[-1].info['max_error'] = 0.0
            atoms_ase[-1].info['n_images'] = n
            atoms_ase[-1].info['algorithm'] = algo
            atoms_ase[-1].info['max_uncertainty'] = 0.0
            atoms_ase[-1].info['converged'] = True
            if len(atoms_ase) == max_steps * n:
                atoms_ase[-1].info['converged'] = False
            write(filename, atoms_ase)
            shutil.copy('./' + filename, results_dir + filename)
            os.remove('./' + filename)

    # 2.B. NEB using CatLearn ################################################

    acquisition_functions = ['acq_1', 'acq_2', 'acq_3']

    for algo in acquisition_functions:

        filename = 'neb_catlearn_' + algo + '_' + str(n) + '_images.traj'

        if not os.path.exists(results_dir + filename):
            neb_catlearn = CatLearnNEB(start='initial_opt.traj',
                                       end='final_opt.traj',
                                       ase_calc=copy.deepcopy(ase_calculator),
                                       n_images=n,
                                       interpolation='idpp',
                                       )

            neb_catlearn.run(fmax=0.05, plot_neb_paths=False,
                             acquisition=algo,
                             unc_convergence=0.100,
                             trajectory=filename)

            # Collect data.
            energy = []
            pred_energy = []
            pred_uncertainty = []
            for i in neb_catlearn.images:
                path_pos_i = i.get_positions()
                structure_i = copy.deepcopy(slab)
                structure_i.positions = path_pos_i
                energy_i = structure_i.get_potential_energy()
                energy.append(energy_i)
                pred_energy.append(i.get_potential_energy())
                pred_uncertainty.append(i.info['uncertainty'])

            energy = np.array(energy)
            pred_energy = np.array(pred_energy)
            pred_unc = np.array(pred_uncertainty)
            max_unc = np.max(pred_unc)
            diff_e_epred = np.abs(energy - pred_energy)
            max_error = np.max(diff_e_epred)
            mean_error = np.mean(diff_e_epred)

            print('Energy (eV): ', energy)
            print('Predicted energy (eV): ', pred_energy)
            print('Uncertainty predicted path: ', pred_unc)
            print('Max 2 sigma (eV):', max_unc)
            print('Error [Abs(Diff(E-Epred))] (eV): ', diff_e_epred)
            print('Max error (eV): ', max_error)
            print('Mean error (eV): ', mean_error)

            # Save results of the run in traj file.
            atoms_ml = read(filename, ':')
            atoms_ml[-1].info['function_calls'] = neb_catlearn.iter
            atoms_ml[-1].info['converged'] = True
            atoms_ml[-1].info['mean_error'] = mean_error
            atoms_ml[-1].info['max_error'] = max_error
            atoms_ml[-1].info['n_images'] = n
            atoms_ml[-1].info['algorithm'] = algo
            atoms_ml[-1].info['max_uncertainty'] = max_unc
            write(filename, atoms_ml)
            shutil.copy('./' + filename, results_dir + filename)
            os.remove('./' + filename)

# Organise results:

results = [['Algorithm', 'Converged',
            'Number of images', 'Function evaluations',
            'Average error', 'Max. error', 'Max. uncertainty']]

for filename in os.listdir(results_dir):
    try:
        print('Reading file:', filename)
        atoms = read(results_dir + filename, ':')
        results.append([atoms[-1].info['algorithm'],
                        atoms[-1].info['converged'],
                        atoms[-1].info['n_images'],
                        atoms[-1].info['function_calls'],
                        atoms[-1].info['mean_error'],
                        atoms[-1].info['max_error'],
                        atoms[-1].info['max_uncertainty']
                        ])
        df = pd.DataFrame(results)

        df.to_csv('results.csv', index=False, header=False)
    except:
        print('Not supported file.')
        pass

df = pd.read_csv('results.csv')

# Plots:

# Colors and markers:
sns.set_style("ticks")
flatui = ["#9b59b6",  "#2ecc71", "#3498db", "#e74c3c", "#34495e","#95a5a6"]
list_markers = itertools.cycle(('D', '>', 's', 'h'))
palette = itertools.cycle(sns.color_palette())

# Axis:
fig = plt.figure(figsize=(5, 8))
ax1 = plt.subplot2grid((3,1), (0,0), rowspan=2)
ax2 = plt.subplot2grid((3,1), (2,0))

# MDMin:
next_color = next(palette)
next_marker = next(list_markers)

df_mdmin = df[df['Algorithm'] == 'MDMin']
df_mdmin = df_mdmin.sort_values('Number of images')
df_mdmin.plot(x='Number of images', y='Function evaluations',
              color=next_color,
              marker=next_marker, markersize=8.0, ls='--',
              markeredgecolor='black', markeredgewidth=0.7,
              label='MDMin', ax=ax1
              )

# FIRE:
next_color = next(palette)
next_marker = next(list_markers)

df_fire = df[df['Algorithm'] == 'FIRE']
df_fire = df_fire.sort_values('Number of images')
df_fire.plot(x='Number of images', y='Function evaluations',
             color=next_color,
             marker=next_marker, markersize=8.0, ls='--',
             markeredgecolor='black', markeredgewidth=0.7,
             label='FIRE', ax=ax1)


# CatLearn NEB. Acquisition 1:
next_color = next(palette)
next_marker = next(list_markers)

df_acq_1 = df[df['Algorithm'] == 'acq_1']
df_acq_1 = df_acq_1.sort_values('Number of images')
df_acq_1.plot(x='Number of images', y='Function evaluations',
             color=next_color,
             marker=next_marker, markersize=8.0, ls='--',
             markeredgecolor='black', markeredgewidth=0.7,
             label='ML-NEB (Acq. 1)', ax=ax1)


df_acq_1.plot(x='Number of images', y='Average error',
             color=next_color,
             marker=next_marker, markersize=8.0, ls='--',
             markeredgecolor='black', markeredgewidth=0.7,
             label='ML-NEB (Acq. 1)', ax=ax2)

# CatLearn NEB. Acquisition 2:
next_color = next(palette)
next_marker = next(list_markers)

df_acq_2 = df[df['Algorithm'] == 'acq_2']
df_acq_2 = df_acq_2.sort_values('Number of images')
df_acq_2.plot(x='Number of images', y='Function evaluations',
             color=next_color,
             marker=next_marker, markersize=8.0, ls='--',
             markeredgecolor='black', markeredgewidth=0.7,
             label='ML-NEB (Acq. 2)', ax=ax1)

df_acq_2.plot(x='Number of images', y='Average error',
             color=next_color,
             marker=next_marker, markersize=8.0, ls='--',
             markeredgecolor='black', markeredgewidth=0.7,
             label='ML-NEB (Acq. 2)', ax=ax2)

# CatLearn NEB. Acquisition 3:
next_color = next(palette)
next_marker = next(list_markers)

df_acq_3 = df[df['Algorithm'] == 'acq_3']
df_acq_3 = df_acq_3.sort_values('Number of images')
df_acq_3.plot(x='Number of images', y='Function evaluations',
             color=next_color,
             marker=next_marker, markersize=8.0, ls='--',
             markeredgecolor='black', markeredgewidth=0.7,
             label='ML-NEB (Acq. 3)', ax=ax1)

df_acq_3.plot(x='Number of images', y='Average error',
             color=next_color,
             marker=next_marker, markersize=8.0, ls='--',
             markeredgecolor='black', markeredgewidth=0.7,
             label='ML-NEB (Acq. 3)', ax=ax2)

# Labels and text:

ax1.set_ylabel('Function evaluations')
ax1.set_xlabel('')
ax1.set_yscale("log", nonposy='clip')
ax1.legend()

ax2.set_ylabel('Average error (eV)')
ax2.set_xlabel('')
ax2.set_ylim([-0.00, 0.05])
ax2.legend().remove()

plt.savefig('./figures/Figure3_Au_NEB.pdf',
                format='pdf',
                dpi=300)
plt.close()



