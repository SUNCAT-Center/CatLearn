import numpy as np
import matplotlib.pyplot as plt
from ase.optimize import BFGS, MDMin, FIRE
from catlearn.optimize.catlearn_neb_optimizer import CatLearnNEB
from ase.calculators.emt import EMT
from ase.neb import NEB
from ase.io import read, write
import copy
import seaborn as sns
import os
import shutil
import pandas as pd

""" 
    Figure 3.C. Number of function calls as a function of number of images. 
    Au heptamer island on Pt(111).
"""


results_dir = './results/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# 1. Structural relaxation. ##################################################

# Setup calculator:
ase_calculator = EMT()

# 1.1. Structures:

slab_initial = read('./initial.traj')
slab_initial.set_calculator(copy.deepcopy(ase_calculator))

slab_final = read('./final.traj')
slab_final.set_calculator(ase_calculator)


# 1.2. Optimize initial and final end-points.

# Initial end-point:
qn = BFGS(slab_initial, trajectory='initial_opt.traj')
qn.run(fmax=0.01)

# Final end-point:
qn = BFGS(slab_final, trajectory='final_opt.traj')
qn.run(fmax=0.01)

# Run NEBS.
n_images = [7, 8, 9, 10, 11, 12, 15, 21, 25, 31]

for n in n_images:

    # 2.A. NEB using ASE ######################################################

    for algo in ['BFGS', 'MDMin', 'FIRE']:

        filename = 'neb_ase_' + algo + '_' + str(n) + '_images.traj'

        if not os.path.exists(results_dir + filename):
            initial_ase = read('initial_opt.traj')
            final_ase = read('final_opt.traj')

            images_ase = [initial_ase]
            for i in range(1, n-1):
                image = initial_ase.copy()
                image.set_calculator(copy.deepcopy(ase_calculator))
                images_ase.append(image)

            images_ase.append(final_ase)

            neb_ase = NEB(images_ase, climb=True)
            neb_ase.interpolate(method='idpp')

            max_steps = 200
            qn_ase = eval(algo)(neb_ase, trajectory=filename)
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
                             unc_convergence=0.050,
                             trajectory=filename)

            # Collect data.
            energy = []
            pred_energy = []
            pred_uncertainty = []
            for i in neb_catlearn.images:
                path_pos_i = i.get_positions()
                structure_i = copy.deepcopy(slab_initial)
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
