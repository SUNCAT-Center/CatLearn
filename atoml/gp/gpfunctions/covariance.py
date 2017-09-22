"""Generation of covariance matrix."""
from __future__ import absolute_import

import numpy as np
from atoml.gp.gpfunctions import kernels as ak


def get_covariance(kernel_dict, matrix1, matrix2=None, regularization=None):
    """Return the covariance matrix of training dataset.

    Parameters
    ----------
    train_matrix : list
        A list of the training fingerprint vectors.
    test_matrix : list
        A list of the test fingerprint vectors.
    kernel_dict : dict of dicts
        A dict containing all data for the kernel functions.
    regularization : None or float
        Smoothing parameter for the Gramm matrix.
    """
    n1, n1_D = np.shape(matrix1)
    if matrix2 is not None:
        n2, n2_D = np.shape(matrix2)
        assert n1_D == n2_D
    else:
        n2 = n1
    # Initialize covariance matrix
    cov = np.zeros([n1, n2])

    # Keep copies of original matrices.
    store1, store2 = matrix1, matrix2

    # Loop over kernels in kernel_dict
    for key in kernel_dict:
        matrix1, matrix2 = store1, store2
        ktype = kernel_dict[key]['type']

        # Select a subset of features for the kernel
        if 'features' in kernel_dict[key]:
            matrix1 = matrix1[:, kernel_dict[key]['features']]
            if matrix2 is not None:
                matrix2 = matrix2[:, kernel_dict[key]['features']]
        theta = ak.kdict2list(kernel_dict[key], n1_D)
        hyperparameters = theta[1]
        if len(theta[0]) == 0:
            scaling = 1
        else:
            scaling = theta[0]

        # Get the covariance matrix
        if 'operation' in kernel_dict[key] and \
           kernel_dict[key]['operation'] == 'multiplication':
            cov *= scaling * eval('ak.' + str(ktype) +
                                  '_kernel(m1=matrix1, m2=matrix2,' +
                                  ' theta=hyperparameters)')
        else:
            cov += scaling * eval('ak.' + str(ktype) +
                                  '_kernel(m1=matrix1, m2=matrix2,' +
                                  ' theta=hyperparameters)')
    if regularization is not None:
        cov += regularization * np.identity(len(cov))
    return cov
