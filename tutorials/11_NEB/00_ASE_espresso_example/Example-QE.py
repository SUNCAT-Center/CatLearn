from ase.io import read
from ase.optimize import BFGS
import shutil
from ase.calculators.espresso import Espresso
from espresso import espresso
from catlearn.optimize.mlneb import MLNEB

""" Example to run ML-NEB using the ASE Espresso calculator.
    Make sure you have set a single-point calculation (for VASP nsw=0).
    The trajectory files for the initial and final end-points are located
    in the 'optimized_structures' directory.

"""

################### ASE Espresso Calculator ##################

convergence = {'energy':1e-5,
               'mixing':0.7,
               'mixing_mode':'local-TF',               
               'maxsteps':10000,
               'diag':'david'
                }
ase_calculator = espresso(
                pw=600,
                dw=6000,
                xc='BEEF',
                kpts=(4, 4, 4),
                nbands=-30,
                sigma=0.05,
                convergence=convergence,
                dipole={'status':False},
                spinpol=False,
                psppath='/home/users/jagt/qe/pseudopotentials/gbrv1.5pbe/',
                output = {'avoidio':False,'removewf':True,'wf_collect':False},
                outdir='ekspressen',
                )

# Optimize initial state:
slab = read('./optimized_structures/initial.traj')
slab.set_calculator(copy.deepcopy(ase_calculator))
# Set magmom to zero:
slab.set_initial_magnetic_moments(np.zeros(len(slab)))

qn = BFGS(slab, trajectory='initial.traj')
qn.run(fmax=0.01)
shutil.copy('./initial.traj', './optimized_structures/initial.traj')

# Optimize final state:
slab = read('./optimized_structures/final.traj')
slab.set_calculator(copy.deepcopy(ase_calculator))
# Set magmom to zero:
slab.set_initial_magnetic_moments(np.zeros(len(slab)))

qn = BFGS(slab, trajectory='final.traj')
qn.run(fmax=0.01)
shutil.copy('./final.traj', './optimized_structures/final.traj')

###### ASE Calculator for CatLearn NEB #######################
ase_calculator = espresso(
                pw=600,
                dw=6000,
                xc='BEEF',
                kpts=(4, 4, 4),
                nbands=-30,
                sigma=0.05,
                convergence=convergence,
                dipole={'status':False},
                spinpol=True,
                psppath='/home/users/jagt/qe/pseudopotentials/gbrv1.5pbe/',
                output = {'avoidio':False,'removewf':True,'wf_collect':False},
                outdir='ekspressen',
                mode='scf' 
                )

####### CatLearn NEB:
neb_catlearn = MLNEB(start='initial.traj', end='final.traj',
                       ase_calc=copy.deepcopy(ase_calculator),
                       n_images=11,
                       )

neb_catlearn.run(fmax=0.05)
