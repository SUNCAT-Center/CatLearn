from ase import Atoms
from gpaw import GPAW
from ase.optimize import *
from ase.optimize.sciopt import *
from catlearn.optimize.catlearn_minimizer import CatLearnMinimizer
import copy
import shutil
import os
from ase.constraints import FixAtoms
from ase.visualize import view
from ase.optimize.sciopt import SciPyFminBFGS
from ase.io import read

""" 
    Benchmark GPAW H2O calculations.
"""

list_kernels = ['SQE_static', 'SQE_isotropic', 'SQE_anisotropic',
                'SQE_sequential']
list_rattle = ['0_00', '0_05', '0_10', '0_15', '0_20', '0_25', '0_30']

for rattle in list_rattle:

    for kernel in list_kernels: # Loop over different default kernels.

        catlearn_version = "_u_1_8_0"
        system = '_rattle' + rattle

        results_dir = './Results/'

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # 0.0 Setup calculator:
        calc = GPAW(nbands=4, h=0.2, mode='lcao', basis='dzp')

        # 1. Structural relaxation. ##################################

        # 1.1. Set up structure:
        a = 6
        b = a / 2
        mol = Atoms('H2O',
                    [(b, 0.7633 + b, -0.4876 + b),
                     (b, -0.7633 + b, -0.4876 + b),
                     (b, b, 0.1219 + b)],
                    cell=[a, a, a])

        mol.rattle(seed=0, stdev=float(rattle.replace('_', '.')))

        # 3. Benchmark.
        ############################################################
        results_dir = './Results/' + system + '/'

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        minimizers = ['BFGS', 'FIRE']

        for i in minimizers:
            filename = i + system + '.traj'
            if not os.path.exists(results_dir + filename):
                initial = mol.copy()
                initial.set_calculator(calc)
                opt = eval(i)(initial, trajectory=filename, )
                opt.run(fmax=0.01, steps=300)
                shutil.copy('./' + filename, results_dir + filename)
                os.remove('./' + filename)

        minimizers = ['BFGS', 'FIRE']

        for i in minimizers:
            filename = i + system + '_' + kernel + catlearn_version
            if not os.path.exists(results_dir + filename + '_catlearn.traj'):
                initial = mol.copy()
                initial.set_calculator(calc)
                opt = CatLearnMinimizer(initial, filename=filename,
                                        ml_calc=kernel)
                opt.run(fmax=0.01, ml_algo=i, max_iter=300)
                shutil.copy('./' + filename + '_catlearn.traj',
                            results_dir + filename + '_catlearn.traj')
                os.remove('./' + filename + '_catlearn.traj')
                os.remove('./' + filename + '_convergence_catlearn.txt')
                os.remove('./warnings_and_errors.txt')
    # 4. Summary of the results:
    ###############################################################################
    from ase.io import read
    import os
    print('\n Summary of the results:\n ------------------------------------')

    # Function evaluations:

    for name_of_file in os.listdir(results_dir):
        try:
            atoms = read(results_dir + name_of_file, ':')
            feval = len(atoms)
            print('Number of function evaluations using ' + name_of_file + ':', feval)
        except:
            pass

# # GENERAL:
from ase.io import read
import os
print('\n Summary of the results:\n ------------------------------------')
    # Function evaluations:
for name_of_file in os.listdir('./'):
        try:
            atoms = read('./' + name_of_file, ':')
            feval = len(atoms)
            print('Number of function evaluations using ' + name_of_file + ':', feval)
        except:
            pass