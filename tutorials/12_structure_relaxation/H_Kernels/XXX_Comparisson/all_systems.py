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
from ase.db import connect
from ase import Atoms


""" 
    Benchmark GPAW benchmark.
"""

list_kernels = [ 'SQE_isotropic', 'SQE_anisotropic', 'SQE_sequential',
                 'SQE_static']

db = connect('./systems.db')
all_structures = db.select()


for structure in all_structures:
    for kernel in list_kernels: # Loop over different default kernels.

        catlearn_version = "_u_2_2_0"
        system = 'system_' + db.get_atoms(structure.id).get_chemical_formula()
        results_dir = './Results/'

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # 0.0 Setup calculator:
        calc = GPAW(mode='lcao', basis='dzp', kpts={'density': 2.0})

        # 1. Structural relaxation. ##################################

        # 1.1. Set up structure:

        mol = db.get_atoms(structure.id)

        # 3. Benchmark.
        ############################################################
        results_dir = './Results/' + system + '/'

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        minimizers = ['BFGS', 'FIRE']

        for i in minimizers:
            filename = i + '_' + system + '.traj'
            if not os.path.exists(results_dir + filename):
                initial = mol.copy()
                initial.set_calculator(calc)
                opt = eval(i)(initial, trajectory=filename)
                opt.run(fmax=0.05, steps=100)
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
                opt.run(fmax=0.05, ml_algo=i, max_iter=100)
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

