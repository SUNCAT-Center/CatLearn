"""Generation of covariance matrix."""
from __future__ import absolute_import

import numpy as np

from catlearn.regression.gpfunctions.kernel_setup import kdict2list
from catlearn.regression.gpfunctions import kernels as ak


def get_covariance(kernel_dict, log_scale, matrix1, matrix2=None,
                   regularization=None, eval_gradients=False):
    """Return the covariance matrix of training dataset.

    Parameters
    ----------
    kernel_dict : dict of dicts
        A dict containing all data for the kernel functions.
    log_scale:
        Flag to define if the hyperparameters are log scale.
    train_matrix : list
        A list of the training fingerprint vectors.
    test_matrix : list
        A list of the test fingerprint vectors.
    regularization : None or float
        Smoothing parameter for the Gramm matrix.
    """
    n1, n1_D = np.shape(matrix1)
    # Make a copy of the feature matrix.
    store1, store2 = matrix1.copy(), None

    if matrix2 is not None:
        assert n1_D == np.shape(matrix2)[1]
        store2 = matrix2.copy()
    cov = None

    # Loop over kernels in kernel_dict
    for key in kernel_dict:
        matrix1 = store1.copy()
        if store2 is not None:
            matrix2 = store2.copy()
        ktype = kernel_dict[key]['type']

        # Select a subset of features for the kernel
        if 'features' in kernel_dict[key]:
            matrix1 = matrix1[:, kernel_dict[key]['features']]
            if matrix2 is not None:
                matrix2 = matrix2[:, kernel_dict[key]['features']]
        theta = kdict2list(kernel_dict[key], n1_D)
        hyperparameters = theta[1]
        if len(theta[0]) == 0:
            scaling = 1.0
        else:
            scaling = theta[0]
        if log_scale:
            scaling = np.exp(scaling)

        # Get mapping from kernel functions.
        k = eval(
            'ak.{}_kernel(m1=matrix1, m2=matrix2, theta=hyperparameters, \
            eval_gradients=eval_gradients, log_scale=log_scale)'.format(ktype))

        # Initialize covariance matrix
        if cov is None:
            cov = np.zeros(np.shape(k))

        # Generate the covariance matrix
        if 'operation' in kernel_dict[key] and \
           kernel_dict[key]['operation'] == 'multiplication':
            cov *= scaling * k
        else:
            cov += scaling * k

    # Apply noise parameter.
    if regularization is not None:
        if log_scale:
            regularization = np.exp(regularization)
        cov += regularization * np.identity(len(cov))

    return cov
