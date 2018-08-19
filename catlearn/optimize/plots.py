import numpy as np
from ase.io import read
import matplotlib.pyplot as plt
from ase.neb import NEBTools

import copy
from ase.atoms import Atoms
from catlearn.optimize.io import array_to_ase


def get_plot_mullerbrown(images, interesting_point, trained_process,
                         list_train):
    """ Function for plotting each step of the toy model Muller-Brown .
    """

    # Generate test datapoints in x and y.
    crange = np.linspace(-0.2, 4.5, 200)
    x_lim = [-1.2, 1.1]
    y_lim = [-0.4, 1.8]

    test_points = 50  # Length of the grid test points (nxn).

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

    # Contour plot for predicted function.

    plt.figure(figsize=(3, 3))
    plt.grid(False)

    # Plot predicted function.

    pred = trained_process.predict(test_fp=test)
    prediction = np.array(pred['prediction'][:, 0])

    zi = plt.mlab.griddata(x, y, prediction, testx, testy, interp='linear')

    plt.contourf(testx, testy, zi, crange, alpha=1.0, cmap='terrain',
                 extend='both')

    # Plot each point evaluated.
    geometry_data = list_train.copy()

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

    plt.xticks([])
    plt.yticks([])

    plt.show()


def get_plot_mullerbrown_p(images, interesting_point, list_train):
    """ Function for plotting each step of the toy model Muller-Brown .
    """

    # Generate test datapoints in x and y.
    crange = np.linspace(-0.2, 4.5, 200)
    x_lim = [-1.2, 1.1]
    y_lim = [-0.4, 1.8]

    test_points = 50  # Length of the grid test points (nxn).

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

    # Contour plot for predicted function.

    plt.figure(figsize=(3, 3))
    plt.grid(False)

    # Plot predicted function.
    prediction = []

    default_structure = copy.deepcopy(images[1])
    calculator = images[1].get_calculator()

    for i in test:
        default_structure = Atoms(default_structure, positions=array_to_ase(
                               i, 1), calculator=calculator)
        prediction.append(default_structure.get_potential_energy())

    zi = plt.mlab.griddata(x, y, prediction, testx, testy, interp='linear')

    plt.contourf(testx, testy, zi, crange, alpha=1.0, cmap='terrain',
                 extend='both')

    # Plot each point evaluated.
    geometry_data = list_train.copy()
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

    plt.xticks([])
    plt.yticks([])

    plt.show()


def get_plots_neb(images, selected=None, iter=None):
    """ Tool for plotting a predicted path for a given set of Atoms objects
        composing an NEB path.

    Parameters
    ----------
    images : Atoms objects or trajectory file.
        Atoms objects or traj file (in ASE format) containing the path to plot.
    selected: int
        Next point to evaluate, chosen by the surrogate model.
    iter: integer
        Iteration number.

    Returns
    -------
    Plot of the NEB path.
    Save results in neb_results.csv file.
    """

    if isinstance(images, str):
        print('Converting trajectory file to Atoms images using ASE...')
        images = read(images, ':')

    neb_tools = NEBTools(images)
    [s, e] = neb_tools.get_fit()[0:2]
    neb_tools.plot_band()

    uncertainty_path = []
    for i in images:
        uncertainty_path.append(i.info['uncertainty'])

    plt.errorbar(s, e, yerr=uncertainty_path,
                 ls='none', ecolor='black', capsize=10.0)
    if iter is not None:
        plt.suptitle('Iteration: {0:.0f}'.format(iter), x=0.85, y=0.9)
    if selected is not None:
        plt.axvline(x=s[selected+1])

    plt.show()
    plt.close()


def get_plot_step(images, interesting_point, trained_process,
                         list_train, scale):
    """ Function for plotting each step of the toy model Muller-Brown .
    """
    iteration = images.info['iteration']
    folder = './plots_ase/'
    plt.figure(figsize=(4.0, 4.0))

    plot_resolution = 150
    limitsx = [-5.5, 0]
    limitsy = [-5.5, 0]
    crange = np.linspace(-0.5, 10.0, 60)
    crange2 = 5

    A = np.linspace(limitsx[0], limitsx[1], plot_resolution)
    B = np.linspace(limitsy[0], limitsy[1], plot_resolution)
    X, Y = np.meshgrid(A, B)
    Z = np.zeros((len(A),len(B)))
    for i in range(0, len(A)):
        for j in range(0,len(B)):
            x = [A[j], B[i], 0]
            pred = trained_process.predict(test_fp=np.array([x]),
                                           uncertainty=True)
            e = np.array(pred['prediction'][:, 0])
            u = np.array(pred['uncertainty'])
            Z[i][j] = e + scale + u
    plt.contourf(X, Y, Z, crange, alpha=1.0, cmap='terrain')
    plt.contour(X, Y, Z, crange2, alpha=0.5, linewidths=1.0, antialiased=True,
                linestyles='dashed')

    plt.scatter(list_train[:, 0], list_train[:, 1], marker='o', c='white', edgecolors='black', alpha=1.0)
    plt.xlim(limitsx[0], limitsx[1])
    plt.ylim(limitsy[0], limitsy[1])
    plt.xticks([])
    plt.yticks([])
    # plt.savefig(fname=(folder+'min_catlearn_himmelblau_iter' + str(iteration) +
    #             '.png'), dpi=500, format='png', transparent=False)

    plt.show()
    plt.close()

def get_plot_step_p(structure, list_train):
    """ Function for plotting each step of the toy model Muller-Brown .
    """

    # Generate test datapoints in x and y.
    crange = np.linspace(-0.5, 10.0, 60)
    x_lim = [-5.5, 0.0]
    y_lim = [-5.5, 0.0]

    test_points = 50  # Length of the grid test points (nxn).

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

    # Contour plot for predicted function.

    plt.figure(figsize=(3, 3))
    plt.grid(False)

    # Plot predicted function.
    prediction = []

    default_structure = copy.deepcopy(structure)
    calculator = structure.get_calculator()

    for i in test:
        default_structure = Atoms(default_structure, positions=array_to_ase(
                               i, 1), calculator=calculator)
        prediction.append(default_structure.get_potential_energy())

    zi = plt.mlab.griddata(x, y, prediction, testx, testy, interp='linear')

    plt.contourf(testx, testy, zi, crange, alpha=1.0, cmap='terrain',
                 extend='both')

    # Plot each point evaluated.
    geometry_data = list_train.copy()
    plt.scatter(geometry_data[:, 0], geometry_data[:, 1], marker='x',
                s=15.0, c='black', alpha=0.8)

    plt.xlim(x_lim[0], x_lim[1])
    plt.ylim(y_lim[0], y_lim[1])

    plt.xticks([])
    plt.yticks([])

    plt.show()
