"""Script to test the prediction functions."""
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
from sys import argv
from atoml.utilities import DescriptorDatabase
from atoml.regression.gpfunctions.kernel_setup import (prepare_kernels,
                                                       kdicts2list, list2kdict)
from atoml.regression.gpfunctions.log_marginal_likelihood import log_marginal_likelihood, dlml
from atoml.preprocess.feature_preprocess import standardize
from atoml.preprocess.scale_target import target_standardize
from scipy.optimize import minimize, basinhopping


wkdir = os.getcwd()
train_size, test_size = 45, 5
n_features = 20
eval_gradients = False
scale_optimizer = False

def get_data():
    """Simple function to pull some training and test data."""
    # Attach the database.
    dd = DescriptorDatabase(db_name='{}/fpv_store.sqlite'.format(wkdir),
                            table='FingerVector')

    # Pull the features and targets from the database.
    names = dd.get_column_names()
    features, targets = names[1:-1], names[-1:]
    feature_data = dd.query_db(names=features)
    target_data = np.reshape(dd.query_db(names=targets),
                             (np.shape(feature_data)[0], ))

    # Split the data into so test and training sets.
    train_features = feature_data[:train_size, :n_features]
    train_targets = target_data[:train_size]
    test_features = feature_data[test_size:, :n_features]
    test_targets = target_data[test_size:]

    return train_features, train_targets, test_features, test_targets


def lml_test(train_features, train_targets, test_features,
             kernel_dict, global_opt=False, algomin='L-BFGS-B', eval_jac=True):
    """Test Gaussian process predictions."""
    # Test prediction routine with linear kernel.
    regularization = 1.e-3
    regularization_bounds = ((1.e-6, None))
    N, N_D = np.shape(train_features)
    kdict, bounds = prepare_kernels(kernel_dict, regularization_bounds,
                                    eval_gradients, N_D)
    print(bounds)
    # Create a list of all hyperparameters.
    theta = kdicts2list(kdict, N_D=N_D)
    theta = np.append(theta, regularization)
    # Define fixed arguments for log_marginal_likelihood
    args = (np.array(train_features), np.array(train_targets),
            kdict, scale_optimizer, eval_gradients, eval_jac)
    # Optimize
    if not global_opt:
        popt = minimize(log_marginal_likelihood, theta,
                        args=args,
                        method=algomin,
                        jac=eval_jac,
                        options={'disp': True},
                        bounds=bounds)
    else:
        minimizer_kwargs = {'method': algomin, 'args': args,
                            'bounds': bounds, 'jac': eval_jac}
        popt = basinhopping(log_marginal_likelihood, theta,
                            minimizer_kwargs=minimizer_kwargs, disp=True)
    return popt


def scale_test(train_matrix, train_targets, test_matrix):
    """Test data scaling functions."""
    sfp = standardize(train_matrix=train_matrix, test_matrix=test_matrix)
    ts = target_standardize(train_targets)
    return sfp['train'], ts['target'], sfp['test']


def lml_plotter(train_features, train_targets, test_features, kernel_dict,
                regularization, d_max=4):
    print('Plotting log marginal likelihood.')
    N, N_D = np.shape(train_features)
    hyperparameters = np.array(kdicts2list(kernel_dict, N_D=N_D))
    hyperparameters = np.append(hyperparameters, regularization)
    for d in range(min(len(hyperparameters), d_max)):
        theta = hyperparameters.copy()
        x0 = hyperparameters[d]
        X = 10 ** np.linspace(np.log10(x0)-3, np.log10(x0)+3, 17)
        Y = []
        dY = []
        for x in X:
            theta[d] = x
            function = log_marginal_likelihood(theta, np.array(train_features),
                                               np.array(train_targets),
                                               kernel_dict, scale_optimizer,
                                               eval_gradients,
                                               eval_jac=True)
            Y.append(function[0])
            dY.append(function[1])
        n_x = np.ceil(np.sqrt(len(theta)))
        n_y = n_x + 1
        ax = plt.subplot(n_x, n_y, d+1)
        ax.semilogx(X, Y, marker='o', linestyle='none')
        for i in range(len(X)):
            dx = X[i] / 10.
            ax.semilogx([X[i]+dx, X[i]-dx],
                        [Y[i]+dY[i][d]*dx, Y[i]-dY[i][d]*dx], c='r')
        ax.axvline(x0)
        ax.set_ylabel('lml')
        ax.set_xlabel('Hyperparameter ' + str(d))

if __name__ == '__main__':
    from pyinstrument import Profiler

    profiler = Profiler()
    profiler.start()
    kernel_dict = {'k1': {'type': 'gaussian', 'width': 0.5, 'scaling': 0.8},
                   'c1': {'type': 'constant', 'const': 1.e-3}}
    train_matrix, train_targets, test_matrix, test_targets = get_data()
    train_features, targets, test_features = scale_test(train_matrix,
                                                        train_targets,
                                                        test_matrix)
    popt_no_jac = lml_test(train_features, targets, test_features,
                           kernel_dict, global_opt=False, eval_jac=False)
    popt_local = lml_test(train_features, targets, test_features,
                          kernel_dict, global_opt=False)
    popt_global = lml_test(train_features, targets, test_features,
                           kernel_dict, global_opt=True)
    kernel_dict = list2kdict(popt_global['x'][:-1], kernel_dict)
    regularization = popt_global['x'][-1]
    profiler.stop()
    print(profiler.output_text(unicode=True, color=True))

# If a user argument 'plot' is passed, display the lml and gradients.
if 'plot' in argv[-1]:
    import matplotlib.pyplot as plt
    lml_plotter(train_features, targets, test_features,
                kernel_dict, regularization)
    plt.tight_layout()
    plt.show()
