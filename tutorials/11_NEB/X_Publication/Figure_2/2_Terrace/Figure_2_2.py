import numpy as np
import matplotlib.pyplot as plt
from ase.optimize import BFGS, MDMin
from catlearn.optimize.catlearn_neb_optimizer import CatLearnNEB
from ase.calculators.emt import EMT
from ase.neb import NEB, NEBTools
from ase.constraints import FixAtoms
import copy
from ase.io import read
import seaborn as sns
import itertools

""" 
    Figure 2.B. Accuracy of the predicted NEB. 
    Diffusion of a Pt atom on an Pt(211) surface.
"""

# Define number of images:
n_images = 11

# 1. Structural relaxation. ##################################################

# Setup calculator:
ase_calculator = EMT()

# 1.2. Optimize initial and final end-points.
slab = read('initial.traj')
slab.set_calculator(copy.deepcopy(ase_calculator))
qn = BFGS(slab, trajectory='initial_opt.traj')
qn.run(fmax=0.01)

slab = read('final.traj')
slab.set_calculator(copy.deepcopy(ase_calculator))
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
    images_ase.append(image)
images_ase.append(final_ase)

neb_ase = NEB(images_ase, climb=True)
neb_ase.interpolate(method='idpp')

qn_ase = MDMin(neb_ase, trajectory='neb_ase.traj')
qn_ase.run(fmax=0.05)

nebtools_ase = NEBTools(images_ase)

aseneb_path = nebtools_ase.get_fit()[0]
aseneb_energy = nebtools_ase.get_fit()[1]
Sf_ase = nebtools_ase.get_fit()[2]
Ef_ase = nebtools_ase.get_fit()[3]

Ef_neb_ase, dE_neb_ase = nebtools_ase.get_barrier(fit=False)

# Generate plots.
sns.set_style("ticks")

# Loop over acquisition functions:
acquisition_functions = ['acq_1', 'acq_2', 'acq_3']
list_palettes = ['Blues_d', 'BuGn_r', 'Reds']

for acq in range(len(acquisition_functions)):

    fig = plt.figure(figsize=(5, 5))
    ax1 = plt.subplot2grid((3,1), (0,0), rowspan=2)
    ax2 = plt.subplot2grid((3,1), (2,0))

    # Plot for different uncertainties.
    list_of_uncertainties = [0.100, 0.075, 0.050, 0.025]

    # Loop colors and markers.

    # A. Custom palette.
    # custom_palette = ['tomato', 'steelblue', 'mediumseagreen', 'gold']
    # palette = itertools.cycle(custom_palette)
    # B. Seaborn palette.
    palette = itertools.cycle(sns.color_palette(list_palettes[acq]))

    # Markers:
    list_markers = ['D', '>', 's', 'h']

    # 2.B. NEB using CatLearn ####################################################

    for loop in range(0, len(list_of_uncertainties)):
        next_color = next(palette)
        neb_catlearn = CatLearnNEB(start='initial_opt.traj',
                                   end='final_opt.traj',
                                   ase_calc=copy.deepcopy(ase_calculator),
                                   n_images=n_images,
                                   interpolation='idpp')

        neb_catlearn.run(fmax=0.05, plot_neb_paths=False,
                         acquisition=acquisition_functions[acq],
                         unc_convergence=list_of_uncertainties[loop])

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
        diff_e_epred = np.abs(energy - pred_energy)

        print('Energy (eV): ', energy)
        print('Predicted energy (eV): ', pred_energy)
        print('Uncertainty predicted path: ', np.array(pred_uncertainty))
        print('Error [Abs(Diff(E-Epred))] (eV): ', diff_e_epred)

        # Plot predicted path.

        unc_string_mev = str(int(1e3 * list_of_uncertainties[loop]))

        ax1.plot(neb_catlearn.sfit, neb_catlearn.efit, color=next_color,
                 linestyle='--', linewidth=1.5)
        ax1.plot(neb_catlearn.s, neb_catlearn.e,
                 color=next_color, alpha=0.9,
                 marker=list_markers[loop], markersize=8.0, ls='',
                 markeredgecolor='black', markeredgewidth=0.7,
                 label=unc_string_mev + ' meV (' + str(neb_catlearn.iter) + ')')

        # Plot error for given predicted path.
        ax2.plot(neb_catlearn.s, diff_e_epred, color=next_color, alpha=0.9,
                 linestyle='-.', linewidth=1.5, markersize=8.0,
                 marker=list_markers[loop],
                 markeredgecolor='black', markeredgewidth=0.7,
                 label=unc_string_mev + ' meV (' + str(neb_catlearn.iter) + ')'
                 )

    # Plot for ASE NEB.
    next_color = next(palette)
    atoms_ase = read('neb_ase.traj', ':')
    n_eval_ase = str(len(atoms_ase) - 2 * n_images)
    ax1.plot(Sf_ase, Ef_ase, ls='-', lw=1.0, marker='', color='black')
    ax1.plot(aseneb_path, aseneb_energy, ls='', marker='o', color='w',
               markeredgecolor='black', markeredgewidth=0.7,
               markersize=7.0, label='ASE' + ' (' + n_eval_ase + ')')

    # Set labels for the plots:
    ax1.legend(loc="upper right")
    ax1.set_ylabel('Energy (eV)')
    ax1.set_xticklabels('')
    ax1.set_ylim([-2.20, 2.20])

    ax2.set_xlabel('Path (Angstrom)')
    ax2.set_ylabel('Error |(E-Epred)| (eV)')
    ax2.set_ylim([-0.030, 0.100])

    plt.savefig('./figures/Terrace_NEB_' + neb_catlearn.acq + '.pdf',
                format='pdf',
                dpi=300)
    plt.close()

