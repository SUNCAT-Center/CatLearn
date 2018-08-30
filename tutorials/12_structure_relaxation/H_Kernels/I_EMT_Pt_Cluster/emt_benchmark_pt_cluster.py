from ase.calculators.emt import EMT
from ase import Atoms
from ase.optimize import *
from ase.optimize.sciopt import *
from ase.optimize.gpmin.gpmin import GPMin
from catlearn.optimize.catlearn_minimizer import CatLearnMinimizer
from ase.optimize.sciopt import *
from ase.io import read
import shutil
import os
import pandas as pd


""" 
    Benchmark EMT Pt cluster calculations.
"""
# Save in csv file:
results = [['Minimizer', 'Rattle', 'Sample', 'Feval', 'Converged', 'Version']]

catlearn_version = "u_2_3_0"

list_rattle = np.linspace(0.1, 0.30, 5)  # Rattle magnitude.
random_samples = np.arange(2)  # Number of random samples.

for sample in random_samples:
    for rattle in list_rattle:

        # 0.0 Setup calculator:
        calc = EMT()

        # 1. Structural relaxation. ##################################

        # 1.1. Set up structure:

        mol = read('./POSCAR')
        mol.rattle(seed=sample, stdev=rattle)

        # 3. Benchmark.
        ############################################################
        results_dir = './Results/'

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        list_minimizers = ['GPMin', 'GPMin_update', 'SciPyFminBFGS', 'FIRE']

        for minimizer in list_minimizers:

            filename = minimizer + '_' + str(rattle) + '_' + str(sample) + \
                       '.traj'
            conv = True

            try:
                if not os.path.exists(results_dir + filename):
                    initial = mol.copy()
                    initial.set_calculator(calc)
                    if minimizer is not 'GPMin_update':
                        opt = eval(minimizer)(initial, trajectory=filename)
                    if minimizer is 'GPMin_update':
                        opt = GPMin(initial, trajectory=filename,
                                    update_hyperparams=True)
                    opt.run(fmax=0.01, steps=200)
                    shutil.copy('./' + filename, results_dir + filename)
                    os.remove('./' + filename)
            except:
                conv = False
                feval = 10000
            # Save results:
            try:
                feval = len(read(results_dir + filename, ':'))
                if feval == 200:
                    conv = False
            except:
                pass
            # Minimizer | Rattle | Sample | Feval | Converged | Version (3.1.6)
            results.append([minimizer, rattle, sample, feval, conv, '3.16.2'])

            # Print and save results:
            print('Function evaluations performed by ' + minimizer + ':', feval)
            df = pd.DataFrame(results)
            df.to_csv('results.csv', index=False, header=False)

        list_kernels = ['SQE_isotropic', 'SQE_sequential']

        for kernel in list_kernels:

            filename = 'catlearn' + '_' + kernel + '_' + str(rattle) + '_' +\
                       str(sample) + '_' + catlearn_version + '.traj'
            conv = True

            if not os.path.exists(results_dir + filename):
                initial = mol.copy()
                initial.set_calculator(calc)
                opt = CatLearnMinimizer(initial, trajectory=filename,
                                        ml_calc=kernel)
                opt.run(fmax=0.01, ml_algo='BFGS', max_iter=200)
                shutil.copy('./' + filename, results_dir + filename)
                os.remove('./' + filename)
            # Save results:
            try:
                feval = len(read(results_dir + filename, ':'))
                if feval == 200:
                    conv = False
            except:
                pass
            # Minimizer | Rattle | Sample | Feval | Converged | Version
            results.append([kernel, rattle, sample, feval,
                            conv, catlearn_version])
            # Print and save results:
            print('Function evaluations performed by ' + kernel + ':', feval)
            df = pd.DataFrame(results)
            df.to_csv('results.csv', index=False, header=False)
