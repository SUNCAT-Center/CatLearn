# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 17:27:47 2017

"""

import numpy as np
from atoml import kernels as ak


def get_covariance(kernel_dict, train_matrix, test_matrix=None,
                   regularization=None):
    """ Returns the covariance matrix of training dataset.

        Parameters
        ----------
        train_matrix : list
            A list of the training fingerprint vectors.

        test_matrix : list
            A list of the test fingerprint vectors.

        kernel_dict : dict of dicts

        regularization : None or float
    """
    N_train, N_D = np.shape(train_matrix)
    if test_matrix is not None:
        N_test, N_D_test = np.shape(test_matrix)
        assert N_D == N_D_test
    else:
        N_test = N_train
    # Initialize covariance matrix
    cov = np.zeros([N_train, N_test])

    # Keep copies of original matrices.
    train_store, test_store = train_matrix, test_matrix

    # Loop over kernels in kernel_dict
    for key in kernel_dict:
        train_matrix, test_matrix = train_store, test_store
        ktype = kernel_dict[key]['type']

        # Select a subset of features for the kernel
        if 'features' in kernel_dict[key]:
            train_matrix = train_matrix[:, kernel_dict[key]['features']]
            if test_matrix is not None:
                test_matrix = test_matrix[:, kernel_dict[key]['features']]
        theta = ak.kdict2list(kernel_dict[key], N_D)

        # Get the covariance matrix
        if 'operation' in kernel_dict[key] and \
           kernel_dict[key]['operation'] == 'multiplication':
            cov *= eval('ak.'+str(ktype) +
                        '_kernel(train_matrix, test_matrix, theta=theta)')
        else:
            cov += eval('ak.' + str(ktype) +
                        '_kernel(train_matrix, test_matrix, theta=theta)')
    if regularization is not None:
        cov += regularization * np.identity(len(cov))
    return cov
