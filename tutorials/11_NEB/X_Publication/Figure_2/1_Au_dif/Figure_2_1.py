import numpy as np
import matplotlib.pyplot as plt
from ase.io import read, write
from ase.optimize import BFGS
from catlearn.optimize.catlearn_neb_optimizer import CatLearnNEB
from ase.calculators.emt import EMT
from ase.io import read
from ase.neb import NEB, NEBTools
from ase.constraints import FixAtoms
from ase.build import fcc100, add_adsorbate
import copy
from catlearn.optimize.constraints import apply_mask
""" 
    Figure 2. Accuracy of the predicted NEB. Diffusion Au atom on Al(111).
"""

# # Define number of images:
n_images = 11

# 1. Structural relaxation. ##################################################

# Setup calculator:
ase_calculator = EMT()

# 1.1. Structures:

# 2x2-Al(001) surface with 3 layers and an
# Au atom adsorbed in a hollow site:
slab = fcc100('Al', size=(2, 2, 3))
add_adsorbate(slab, 'Au', 1.7, 'hollow')
slab.center(axis=2, vacuum=4.0)
slab.set_calculator(copy.deepcopy(ase_calculator))

# Fix second and third layers:
mask = [atom.tag > 1 for atom in slab]
slab.set_constraint(FixAtoms(mask=mask))

# 1.2. Optimize initial and final end-points.

# Initial end-point:
qn = BFGS(slab, trajectory='initial_opt.traj')
qn.run(fmax=0.01)

# Final end-point:
slab[-1].x += slab.get_cell()[0, 0] / 2
qn = BFGS(slab, trajectory='final_opt.traj')
qn.run(fmax=0.01)

# 2.A. NEB using ASE #########################################################

initial_ase = read('initial_opt.traj')
final_ase = read('final_opt.traj')
constraint = FixAtoms(mask=[atom.tag > 1 for atom in initial_ase])

images_ase = [initial_ase]
for i in range(1, n_images-1):
    image = initial_ase.copy()
    image.set_calculator(copy.deepcopy(ase_calculator))
    image.set_constraint(constraint)
    images_ase.append(image)

images_ase.append(final_ase)

neb_ase = NEB(images_ase, climb=True)
neb_ase.interpolate(method='idpp')

qn_ase = BFGS(neb_ase, trajectory='neb_ase.traj')
qn_ase.run(fmax=0.05)

nebtools_ase = NEBTools(images_ase)

aseneb_path = nebtools_ase.get_fit()[0]
aseneb_energy = nebtools_ase.get_fit()[1]
Sf_ase = nebtools_ase.get_fit()[2]
Ef_ase = nebtools_ase.get_fit()[3]

Ef_neb_ase, dE_neb_ase = nebtools_ase.get_barrier(fit=False)
nebtools_ase.plot_band()

# plt.show()

# 2.B. NEB using CatLearn ####################################################

neb_catlearn = CatLearnNEB(start='initial_opt.traj',
                           end='final_opt.traj',
                           ase_calc=copy.deepcopy(ase_calculator),
                           n_images=n_images,
                           interpolation='idpp')

neb_catlearn.run(fmax=0.05, plot_neb_paths=False)

# Difference between real value of the energy and the predicted one.

energy = []
pred_energy = []
pred_uncertainty = []
for i in neb_catlearn.images:
    path_pos_i = i.get_positions()
    structure_i = copy.deepcopy(slab)
    structure_i.positions = path_pos_i
    energy_i = structure_i.get_potential_energy()
    energy.append(energy_i)
    pred_energy.append(i.get_potential_energy())
    pred_uncertainty.append(i.info['uncertainty'])

energy = np.array(energy)
pred_energy = np.array(pred_energy)
diff_e_epred = energy - pred_energy

print('Energy (eV): ', energy)
print('Predicted energy (eV): ', pred_energy)
print('Uncertainty predicted path (2 var): ', np.array(pred_uncertainty))
print('Error [Abs(Diff(E-Epred))] (eV): ', diff_e_epred)

# Plots.

# Figure A.

plt.plot(neb_catlearn.sfit, neb_catlearn.efit, color='black',
             linestyle='--', linewidth=1.5)
plt.plot(neb_catlearn.s, neb_catlearn.e,
         color='red', alpha=0.5,
         marker='o', markersize=10.0, ls='',
         markeredgecolor='black', markeredgewidth=0.9)
plt.xlabel('Path (Angstrom)')
plt.ylabel('Energy (eV)')
plt.tight_layout(h_pad=1)
plt.savefig('./figures/1_Au_pred_neb.pdf', format='pdf', dpi=300)
plt.close()

# Figure B.
plt.plot(neb_catlearn.s, diff_e_epred, color='black',
             linestyle='--', linewidth=1.5, markersize=10.0)
plt.xlabel('Path (Angstrom)')
plt.ylabel('Error |(E-Epred)| (eV)')
plt.tight_layout(h_pad=1)
plt.savefig('./figures/1_Au_diff_e_epred.pdf', format='pdf', dpi=300)
plt.close()