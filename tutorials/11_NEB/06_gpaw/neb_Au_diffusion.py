from ase.build import fcc100, add_adsorbate
from ase.io import read
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from ase.parallel import rank, parprint
from gpaw import GPAW, FermiDirac

from catlearn.optimize.mlneb import MLNEB


"""
    Toy model for the diffusion of a Au atom on an Al(111) surface.
    This example contains:
    1. Optimization of the initial and final end-points of the reaction path.
    2.A. NEB optimization using CI-NEB as implemented in ASE.
    2.B. NEB optimization using our machine-learning surrogate model.
    3. Comparison between the ASE NEB and our ML-NEB algorithm.
"""

# 1. Structural relaxation.

# Setup calculator:
calc_args = {'mode': 'lcao',
             'h': 0.18,
             'basis': 'dzp',
             'symmetry': 'off',
             'xc': 'PBE',
             'txt': open('gpaw.txt', 'a'),
             'occupations': FermiDirac(0.03)}

# 1.1. Structures:

# 2x2-Al(001) surface with 3 layers and an
# Au atom adsorbed in a hollow site:
slab = fcc100('Al', size=(2, 2, 3))
add_adsorbate(slab, 'Au', 1.7, 'hollow')
slab.center(axis=2, vacuum=4.0)
slab.set_calculator(GPAW(**calc_args))

# Fix second and third layers:
mask = [atom.tag > 1 for atom in slab]
slab.set_constraint(FixAtoms(mask=mask))

# 1.2. Optimize initial and final end-points.

# Initial end-point:
qn = BFGS(slab, trajectory='initial.traj')
qn.run(fmax=0.01)

# Final end-point:
slab[-1].x += slab.get_cell()[0, 0] / 2
qn = BFGS(slab, trajectory='final.traj')
qn.run(fmax=0.01)

# # Define number of images:
n_images = 7

# 2.B. NEB using CatLearn
if rank == 0:
    neb_catlearn = MLNEB(start='initial.traj',
                         end='final.traj',
                         ase_calc=GPAW(**calc_args),
                         n_images=n_images,
                         interpolation='idpp', restart=False)

    neb_catlearn.run(fmax=0.05, trajectory='ML-NEB.traj')

# 3. Summary of the results #################################################
# ML-NEB:
atoms_catlearn = read('evaluated_structures.traj', ':')
n_eval_catlearn = len(atoms_catlearn) - 2
parprint('Number of function evaluations CatLearn:', n_eval_catlearn)
