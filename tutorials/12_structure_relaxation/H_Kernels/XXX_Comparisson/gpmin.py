from ase import Atoms
# from gpaw import GPAW
from ase.optimize import *
from ase.optimize.sciopt import *
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

db = connect('./systems.db')
all_structures = db.select()

# 0.0 Setup calculator:
# calc = GPAW(mode='lcao', basis='dzp', kpts={'density': 2.0})

# structure = all_structures[1]

# mol = db.get_atoms(structure.id)
#
# mol.set_calculator(calc)
#
# view(mol)
#
