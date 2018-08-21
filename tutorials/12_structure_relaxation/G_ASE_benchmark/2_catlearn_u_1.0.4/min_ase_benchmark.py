from ase import Atoms
from gpaw import GPAW
from ase.optimize import BFGS, FIRE, MDMin
from ase.optimize.sciopt import *
from catlearn.optimize.catlearn_minimizer import CatLearnMinimizer
from ase.io import read, write
import copy
from ase.visualize import view
from ase.db import connect
import ase.db


# 0. Setup calculator:
###############################################################################
ase_calculator = GPAW(mode='lcao',
     basis='dzp',
     kpts={'density': 2.0})

# 1 Structure:
db = ase.db.connect('systems.db')
common_initial = db.get_atoms('Au8CO')


np.random.seed(1)
for i in common_initial:
    if i.position[2] > 8.50:
        i.position = i.position + np.random.normal(scale=0.0)

# 2.B. Optimize structure using CatLearn:
###############################################################################

initial_catlearn = copy.deepcopy(common_initial)
initial_catlearn.set_calculator(ase_calculator)

catlearn_opt = CatLearnMinimizer(initial_catlearn, filename='results')
catlearn_opt.run(fmax=0.02, ml_algo='SciPyFminCG')

