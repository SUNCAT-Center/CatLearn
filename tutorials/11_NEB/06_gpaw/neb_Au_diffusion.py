from ase.build import fcc100, add_adsorbate
from ase.io import read
from ase.constraints import FixAtoms
from ase.neb import NEB
from ase.optimize import BFGS

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
             'xc': 'PBE',
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

# 2.A. NEB using ASE

initial_ase = read('initial.traj')
final_ase = read('final.traj')
constraint = FixAtoms(mask=[atom.tag > 1 for atom in initial_ase])

images_ase = [initial_ase]
for i in range(1, n_images-1):
    image = initial_ase.copy()
    image.set_calculator(GPAW(**calc_args))
    image.set_constraint(constraint)
    images_ase.append(image)

images_ase.append(final_ase)

neb_ase = NEB(images_ase, climb=True)
neb_ase.interpolate(method='idpp')

qn_ase = BFGS(neb_ase, trajectory='neb_ase.traj')
qn_ase.run(fmax=0.05)

# 2.B. NEB using CatLearn

neb_catlearn = MLNEB(start='initial.traj',
                     end='final.traj',
                     ase_calc=GPAW(**calc_args),
                     n_images=n_images,
                     interpolation='idpp', restart=False)

neb_catlearn.run(fmax=0.05, trajectory='ML-NEB.traj')

# 3. Summary of the results #################################################

# NEB ASE:
print('\nSummary of the results: \n')

atoms_ase = read('neb_ase.traj', ':')
n_eval_ase = int(len(atoms_ase) - 2 * (len(atoms_ase)/n_images))

print('Number of function evaluations CI-NEB implemented in ASE:', n_eval_ase)

# ML-NEB:
atoms_catlearn = read('evaluated_structures.traj', ':')
n_eval_catlearn = len(atoms_catlearn) - 2
print('Number of function evaluations CatLearn:', n_eval_catlearn)

# Comparison:
print('\nThe ML-NEB algorithm required ',
      (n_eval_ase/n_eval_catlearn),
      'times less number of function evaluations than '
      'the standard NEB algorithm.')
