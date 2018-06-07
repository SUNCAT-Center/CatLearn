import numpy as np
from ase.io import read
from ase import Atoms
import matplotlib.pyplot as plt
from ase.neb import NEBTools


def plot_each_step(self):
    plt.figure(figsize=(5,5))
    ### Real potential energy surface grid search ##############
    slab_real = self.ase_ini
    slab_real.set_calculator(self.ase_calc)

    xlimits = [14.0, 15.25]

    distances = np.linspace(xlimits[0], xlimits[1], 100)
    list_of_distances = []
    list_of_energies = []

    for d in distances:
        pos = slab_real.get_positions()
        pos[-1] = [0.0, 0.0, d]
        slab_real.set_positions(newpositions=pos)
        list_of_distances = np.append(list_of_distances, [d])
        list_of_energies = np.append(list_of_energies,
        [slab_real.get_potential_energy()])
        # print(self.ml_calc.gp)
        # exit()
    ####################################################################

    ######## TRAINED POINTS ############
    list_trained_points = []
    for i in range(0,len(self.list_train)):
        list_trained_points = np.append(list_trained_points,
        (self.list_train[i][-1]))


    ############## PREDICTED MEAN AND UNC ###########################

    test = np.array([distances])
    test = np.reshape(test, (len(test[0]), 1))
    pred = self.trained_process.predict(test_fp=test, uncertainty=True)

    prediction = np.array(pred['prediction'][:, 0])
    # Calculate the uncertainty of the predictions.
    uncertainty = np.array(pred['uncertainty'])

    # Get confidence interval on predictions.
    upper = prediction + uncertainty
    lower = prediction - uncertainty

    plt.plot(list_of_distances,list_of_energies)
    plt.plot(test, prediction)
    plt.scatter(list_trained_points, self.list_targets[:,0])

    plt.fill_between(distances, upper, lower, interpolate=True,
                    color='blue', alpha=0.2)

    plt.axvline(self.interesting_point[-1][-1])
    plt.xlabel('Height (Angstrom)')
    plt.ylabel('Energy (eV)')
    plt.xlim([xlimits[0],xlimits[1]])
    plt.ylim([16.8, 18.5])
    plt.show()


def plot_neb_mullerbrown(images, interesting_point, trained_process,
                         list_train):
    # Generate test datapoints in x and y.

    crange = np.linspace(-0.5, 5.0, 200)
    x_lim = [-1.2, 1.1]
    y_lim = [-0.4, 1.8]

    test_points = 50 # Length of the grid test points (nxn).

    test = []
    testx = np.linspace(x_lim[0], x_lim[1], test_points)
    testy = np.linspace(y_lim[0], y_lim[1], test_points)
    for i in range(len(testx)):
        for j in range(len(testy)):
            test1 = [testx[i],testy[j], 0.0]
            test.append(test1)
    test = np.array(test)

    x = []
    for i in range(len(test)):
        t = test[i][0]
        t = x.append(t)
    y = []
    for i in range(len(test)):
        t = test[i][1]
        t = y.append(t)

    # Contour plot for predicted function.

    plt.figure(figsize=(3, 3))
    plt.grid(False)

    # Plot predicted function.

    geometry_data = list_train.copy()
    pred = trained_process.predict(test_fp=test)
    prediction = np.array(pred['prediction'][:, 0])

    zi = plt.mlab.griddata(x, y, prediction, testx, testy,
                           interp='linear')

    plt.contourf(testx, testy, zi, crange, alpha=1.0, cmap='terrain')

    # Plot each point evaluated (Muller).

    plt.scatter(geometry_data[:, 0], geometry_data[:, 1], marker='x',
                s=15.0, c='black', alpha=0.8)

    # Plot NEB/GP optimized.
    for i in images:
        pos_i_x = i.get_positions()[0][0]
        pos_i_y = i.get_positions()[0][1]
        plt.scatter(pos_i_x, pos_i_y, marker='o', s=8.0,
                c='red', edgecolors='red', alpha=0.9)
    plt.scatter(interesting_point[0], interesting_point[1],
                marker='o', c='white', edgecolors='black')

    plt.xlim(x_lim[0], x_lim[1])
    plt.ylim(y_lim[0], y_lim[1])

    plt.show()


def plot_predicted_neb_path(images, climb_image=None, filename=''):
    """ Tool for plotting a predicted path of a given trajectory file.

    Parameters
    ----------
    images: Atoms objects containing the path to plot.

    Returns
    -------
    Plot Save image in pdf file and
    """

    iter = None

    iter = images[0].info['iteration']
    accepted_path = images[0].info['accepted_path']

    neb_tools = NEBTools(images)
    [s, E, Sfit, Efit, lines] = neb_tools.get_fit()

    energies_pred_neb = []

    energy_img_0 = images[0].get_total_energy()


    uncertainties_pred_neb = []
    for i in images:
        energies_pred_neb.append(i.get_total_energy() - energy_img_0)
        uncertainties_pred_neb.append(i.info['uncertainty'])

    plt.plot(Sfit, Efit,'--', linewidth= 1.2, color='black')

    plt.plot(s, E, 'o', markersize=11.0, alpha=0.4, color='navy')
    plt.errorbar(s[1:-1], energies_pred_neb[1:-1],
                 yerr=uncertainties_pred_neb[1:-1], fmt='o', markersize=4,
                 color='lightgray',
                 alpha=1.0, ecolor='black', elinewidth=1.5, capsize=5,
                 capthick=1.5)

    plt.plot(s[0], E[0], 'o', markersize=11.0, alpha=1.0,
                color='lightgray')
    plt.plot(s[-1], E[-1], 'o', markersize=11.0, alpha=1.0,
                color='lightgray')

    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.xlabel('Reaction path [$\AA$]')
    plt.ylabel('Energy [eV]')
    Ef_neb = max(Efit) - E[0]
    Er_neb = max(Efit) - E[-1]
    dE_neb = E[-1] - E[0]

    plt.title('Iter: {0:.0f}; E$_f$: {1:.3f} eV; E$_r$: {2:.3f} eV'.format(
        iter, Ef_neb, Er_neb) + '; Accepted:' + str(accepted_path) + '; CI:' +
        str(climb_image))

    if accepted_path is None:
        plt.title('Iter: {0:.0f}; E$_f$: {1:.3f} eV; E$_r$: {2:.3f} '
        'eV; Max. uncertainty: {3:.3f} eV'.format(
        iter, Ef_neb, Er_neb, np.max(uncertainties_pred_neb)))


    # plt.savefig(fname=filename + 'reaction_path_iteration_' + str(iter)
    #                 +'.pdf', dpi=300, format='pdf', transparent=True)
    plt.show()
    plt.close()

