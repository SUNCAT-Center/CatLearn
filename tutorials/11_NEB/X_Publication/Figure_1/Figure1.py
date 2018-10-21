import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from catlearn.optimize.functions_calc import MullerBrown
from ase import Atoms
from ase.optimize import BFGS
from catlearn.optimize.catlearn_neb_optimizer import CatLearnNEB
import seaborn as sns
sns.set_style("ticks")

"""
    Figure 1. Acquisition functions (Muller-Brown potential).
"""

# Define number of images for the NEB:
n_images = 11

def get_plots_neb(catlearn_neb):

    """ Function for plotting each step of the toy model Muller-Brown .
    """

    fig = plt.figure(figsize=(5, 8))
    ax1 = plt.subplot2grid((7, 1), (0, 0), rowspan=5)
    ax2 = plt.subplot2grid((7, 1), (5, 0), rowspan=2)

    # Grid test points (x,y).
    x_lim = [-1.4, 1.3]
    y_lim = [-0.6, 2.0]

    # Axes.
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    ax2.set_xlabel('Path distance (Angstrom)')
    ax2.set_ylabel('Function value (a.u.)')
    ax1.set_xlim(x_lim)
    ax1.set_ylim(y_lim)

    # Range of energies:
    max_color = +0.75
    min_color = -1.75

    # Length of the grid test points (nxn).
    test_points = 250

    test = []
    testx = np.linspace(x_lim[0], x_lim[1], test_points)
    testy = np.linspace(y_lim[0], y_lim[1], test_points)
    for i in range(len(testx)):
        for j in range(len(testy)):
            test1 = [testx[i], testy[j], 0.0]
            test.append(test1)
    test = np.array(test)

    x = []
    for i in range(len(test)):
        t = test[i][0]
        x.append(t)
    y = []
    for i in range(len(test)):
        t = test[i][1]
        y.append(t)

    # Plot predicted function.

    pred = catlearn_neb.gp.predict(test_fp=test)
    prediction = np.array(pred['prediction'][:, 0]) + catlearn_neb.max_target

    crange = np.linspace(min_color, max_color, 1000)

    zi = plt.mlab.griddata(x, y, prediction, testx, testy, interp='linear')

    image = ax1.contourf(testx, testy, zi, crange, alpha=1., cmap='Spectral_r',
                        extend='both', antialiased=False)
    for c in image.collections:
        c.set_edgecolor("face")
        c.set_linewidth(0.000001)

    crange2 = np.linspace(min_color, max_color, 10)
    ax1.contour(testx, testy, zi, crange2, alpha=0.6, antialiased=True)

    interval_colorbar = np.linspace(min_color, max_color, 5)

    fig.colorbar(image, ax=ax1, ticks=interval_colorbar, extend='None',
                 panchor=(0.5,0.0),
                 orientation='horizontal', label='Function value (a.u.)')

    # Plot each point evaluated.
    geometry_data = catlearn_neb.list_train.copy()

    ax1.scatter(geometry_data[:, 0], geometry_data[:, 1], marker='x',
                s=50.0, c='black', alpha=1.0)

    # Plot NEB/GP optimized.
    for i in catlearn_neb.images:
        pos_i_x = i.get_positions()[0][0]
        pos_i_y = i.get_positions()[0][1]
        ax1.scatter(pos_i_x, pos_i_y, marker='o', s=20.0,
                    c='red', edgecolors='red', alpha=1.)
    ax1.scatter(catlearn_neb.interesting_point[0],
                catlearn_neb.interesting_point[1], s=70.,
                marker='o', c='white', edgecolors='black')

    plt.tight_layout(h_pad=1)

    # Text in box:
    t = ax1.text(0.9, 0.87, str(catlearn_neb.iter),
                 transform=ax1.transAxes, fontsize=25)
    t.set_bbox(dict(facecolor='white', alpha=1.0, edgecolor='black'))

    if catlearn_neb.argmax_unc is not None:
        ax2.axvline(x=catlearn_neb.s[int(catlearn_neb.argmax_unc)+1],
                    color='yellow', linewidth=20.0, alpha=0.3)
    ax2.plot(catlearn_neb.sfit, catlearn_neb.efit, color='black',
             linestyle='--', linewidth=1.5)
    ax2.errorbar(catlearn_neb.s, catlearn_neb.e,
                 yerr=catlearn_neb.uncertainty_path,
                 markersize=0.0, ecolor='darkslateblue',
                 ls='', elinewidth=2.0, capsize=1.0)
    ax2.plot(catlearn_neb.s, catlearn_neb.e,
             color='red', alpha=0.5,
             marker='o', markersize=10.0, ls='',
             markeredgecolor='black', markeredgewidth=0.9)

# 1. Structural relaxation. ##################################################

# Setup calculator.
calc = MullerBrown()

# 1.1. Structures.
initial_structure = Atoms('C', positions=[(-0.55, 1.41, 0.0)])
final_structure = Atoms('C', positions=[(0.626, 0.025, 0.0)])

initial_structure.set_calculator(calc)
final_structure.set_calculator(calc)

# 1.2. Optimize initial and final end-points.

# 1.2.1. Initial end-point.
initial_opt = BFGS(initial_structure, trajectory='initial_optimized.traj')
initial_opt.run(fmax=0.01)

# 1.2.2 Final end-point:
final_opt = BFGS(final_structure, trajectory='final_optimized.traj')
final_opt.run(fmax=0.01)


# 2. Plot Muller step for each acquisition function.

# Define steps and acquisition functions to plot.
steps_plots = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
acquisition_functions = ['acq_1', 'acq_2', 'acq_3']

for acq in acquisition_functions:
    for max_steps in steps_plots:
        initial = read('initial_optimized.traj')
        final = read('final_optimized.traj')

        catlearn_neb = CatLearnNEB(start='initial_optimized.traj',
                                   end='final_optimized.traj',
                                   ase_calc=calc,
                                   n_images=n_images,
                                   interpolation='linear', restart=False)

        catlearn_neb.run(fmax=0.05, steps=max_steps, acquisition=acq,
                         unc_convergence=0.100)
        get_plots_neb(catlearn_neb)
        plt.savefig('./figures/pred_NEB_' + acq + '_iter_' + str(
                    max_steps) + '.pdf', format='pdf', dpi=300)
        plt.close()
