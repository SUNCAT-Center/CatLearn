from ase.io import read
from ase.visualize import view
from ase.neb import NEBTools
import matplotlib.pyplot as plt
import numpy as np


def plotneb(trajectory='ML_NEB_catlearn.traj', view_path=True):

    """
    Plot NEB path from a trajectory file containing the optimized images.
    This is meant to be used with ML-NEB.
    The error bars show the uncertainty for each image along the path.
    """

    images = read(trajectory, ':')

    # Get fit from NEBTools.
    nebtools = NEBTools(images)
    nebfit = nebtools.get_fit()

    x_pred = nebfit[0]
    e_pred = nebfit[1]

    x_fit = nebfit[2]
    e_fit = nebfit[3]

    e_barrier = np.max(e_pred)
    print('Energy barrier:', e_barrier, 'eV')

    u_pred = []
    for i in images:
        u_pred.append(i.info['uncertainty'])

    fig, ax = plt.subplots(figsize=(8, 5))

    prop_plots = dict(arrowstyle="<|-|>, head_width=0.5, head_length=1.",
                      connectionstyle="arc3", color='teal',
                      ls='-', lw=0.0)
    ax.annotate("", xy=(x_pred[np.argmax(e_pred)], np.max(e_pred)),
                xycoords='data', xytext=(x_pred[np.argmax(e_pred)], 0.0),
                textcoords='data', arrowprops=prop_plots)

    prop_plots = dict(arrowstyle="|-|", connectionstyle="arc3",
                      ls='-', lw=2., color='teal')
    ax.annotate("", xy=(x_pred[np.argmax(e_pred)], np.max(e_pred)),
                xycoords='data', xytext=(x_pred[np.argmax(e_pred)], 0.0),
                textcoords='data', arrowprops=prop_plots)

    ax.annotate(s=str(np.round(e_barrier, 3))+' eV',
                xy=(x_pred[np.argmax(e_pred)], np.max(e_pred)/1.65),
                xycoords='data',
                fontsize=15.0,
                textcoords='data', ha='right', rotation=90,
                color='teal')

    ax.plot(x_fit, e_fit, color='black', linestyle='--', linewidth=1.5)

    ax.errorbar(x_pred, e_pred, yerr=u_pred, alpha=0.8,
                markersize=0.0, ecolor='midnightblue',
                ls='', elinewidth=3.0, capsize=1.0)

    ax.plot(x_pred, e_pred,
            color='firebrick', alpha=0.7,
            marker='o', markersize=13.0, markeredgecolor='black', ls='')

    ax.set_xlabel('Path distance ($\AA$)')
    ax.set_ylabel('Energy (eV)')
    plt.tight_layout(h_pad=1)

    print('Saving pdf file with the NEB profile in file: ' + 'MLNEB.pdf')
    plt.savefig('./' 'MLNEB.pdf', format='pdf')
    plt.show()

    if view_path is True:
        print('Visualizing NEB images in ASE...')
        view(images)
