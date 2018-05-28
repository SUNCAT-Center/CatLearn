from ase.build import fcc100, add_adsorbate
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.optimize import QuasiNewton
from ase.io import read, write
from ase.constraints import FixAtoms
from ase.neb import NEB
from ase.optimize import LBFGS, FIRE, BFGS, BFGSLineSearch, MDMin
from ase.visualize import view
import matplotlib.pyplot as plt
from catlearn.optimize.catlearn_neb_optimizer import NEBOptimizer
from ase.neb import NEBTools
import copy
from ase.db import connect


# 0. Connect to database #
selected_structure = 1
db_path = '/Users/jagt/Database/NEB-pairings-fixedbase.db'
db = connect(db_path)

# 1. Generate calculator from database #######
calc_dict = db.get(selected_structure).data
calc_dict['psppath'] = '/home/users/vossj/suncat/psp/gbrv1.5pbe'
calc_dict['outdir'] = 'initial'
calc_dict['mode'] = 'ase3'
calc_dict['opt_algorithm'] = 'ase3'
del(calc_dict['opt_algorithm'])

# ase_calculator = espresso(**calc_dict)

# 2. Select NEB path from database ####

neb_path = list(db.select('neb_id=' + str(selected_structure)))

initial_slab = neb_path[0].toatoms()
final_slab = neb_path[-1].toatoms()

path_slab = []
for i in range(0, len(neb_path)):
    atoms_i = neb_path[i].toatoms()
    path_slab.append(atoms_i)

write('initial.traj', initial_slab)
write('final.traj', final_slab)
write('path_neb.traj', path_slab)

a = read('initial.traj')
b = read('final.traj')
c = read('path_neb.traj', ':')


