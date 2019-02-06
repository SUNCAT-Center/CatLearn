import numpy as np
from espresso import espresso
from ase.visualize import view
from catlearn.optimize.mlneb import MLNEB
from catlearn.optimize.mlmin import MLMin
import copy
from ase.io import read, write
import shutil

xc = 'BEEF'

# Setup ASE calculator (e.g. QE).
calc = espresso(mode='scf',
                outdir='ekspressen',
                pw=500.,
                dw=5000.,
                xc=xc, 
                kpts=(4,4,1),
                nbands=-20,
                occupations = 'smearing' ,
                sigma = 0.1,
                smearing = 'gaussian' ,
                parflags='-npool 2',
                convergence={'energy':1e-5,'mixing':0.25,'maxsteps':300},
                dipole={'status':True},
                spinpol=False)

# Optimize the initial and final states using a minimizer (MLMin, BFGS, FIRE ...)

# Optimize the initial state using MLMin:
slab = read('./structures/initial.traj')
slab.set_calculator(copy.deepcopy(calc))
qn = MLMin(slab, trajectory='initial.traj')
qn.run(fmax=0.03)
shutil.copy('./initial.traj', './structures/initial.traj')

# Optimize final state using MLMin:
slab = read('./structures/final.traj')
slab.set_calculator(copy.deepcopy(calc))
qn = MLMin(slab, trajectory='final.traj')
qn.run(fmax=0.03)
shutil.copy('./final.traj', './structures/final.traj')

# Perform a Minimum Energy Path (MEP) search using ML-NEB:
neb_catlearn = MLNEB(start='./structures/initial.traj', # Initial end-point.
                     end='./structures/final.traj', # Final end-point.
                     ase_calc=copy.deepcopy(calc), # Calculator, it must be the same as the one used for the optimizations.
                     n_images=15, # Number of images (interger or float, see above).
                     interpolation='idpp', # Choose between linear or idpp interpolation (as implemented in ASE). You can also feed a list of Atoms with your own interpolation.
                     restart=True
                     )                     
neb_catlearn.run(fmax=0.05, trajectory='ML-NEB.traj')
