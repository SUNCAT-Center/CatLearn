from ase.io import read
from ase.optimize import BFGS
from ase.calculators.vasp import Vasp
import shutil
import copy
from catlearn.optimize.mlneb import MLNEB

""" Example to run ML-NEB using the VASP calculator.
    Make sure you have set a single-point calculation (for VASP nsw=0).
    The trajectory files for the initial and final end-points are located
    in the 'optimized_structures' directory.
    
"""

################### VASP Calculator ##################
ase_calculator = Vasp(encut=400, # 400 for surfaces.
                        xc='PBE',
                        gga='PE',
                        istart = 0,
                        lwave=False,
                        lcharg=False,
                        kpts  = (1, 1, 1),
                        ivdw = 12,
                        gamma = True, # Gamma-centered (defaults to Monkhorst-Pack)
                        ismear=0,
                        sigma = 0.1,
                        algo = 'Normal',
                        ibrion=-1,
                        ediffg=-0.01,  # forces
                        ediff=1e-5,  #energy conv.
                        prec='Accurate', # Slab 
                        nsw=0, # don't use the VASP internal relaxation, only use ASE
                        ispin=1,
                        nelm=300
                        )

# Optimize initial state:
slab = read('./optimized_structures/initial.traj')
slab.set_calculator(copy.deepcopy(ase_calculator))
qn = BFGS(slab, trajectory='initial.traj')
qn.run(fmax=0.01)
shutil.copy('./initial.traj', './optimized_structures/initial.traj')

# Optimize final state:
slab = read('./optimized_structures/final.traj')
slab.set_calculator(copy.deepcopy(ase_calculator))
qn = BFGS(slab, trajectory='final.traj')
qn.run(fmax=0.01)
shutil.copy('./final.traj', './optimized_structures/final.traj')

####### CatLearn NEB:

neb_catlearn = MLNEB(start='initial.traj', end='final.traj',
                     ase_calc=copy.deepcopy(ase_calculator),
                     n_images=11
                     )

neb_catlearn.run(fmax=0.05)


