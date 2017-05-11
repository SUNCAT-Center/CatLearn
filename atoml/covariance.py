# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 17:27:47 2017

"""

import numpy as np
from atoml import kernels as ak


def get_covariance(kernel_dict, matrix1, matrix2=None,
                   regularization=None):
    """ Returns the covariance matrix of training dataset.

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
    N1, N1_D = np.shape(matrix1)
    if matrix2 is not None:
        N2, N2_D = np.shape(matrix2)
        assert N1_D == N2_D
    else:
        N2 = N1
    # Initialize covariance matrix
    cov = np.zeros([N1, N2])

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
        theta = ak.kdict2list(kernel_dict[key], N1_D)

        # Get the covariance matrix
        if 'operation' in kernel_dict[key] and \
           kernel_dict[key]['operation'] == 'multiplication':
            cov *= eval('ak.'+str(ktype) +
                        '_kernel(matrix1, matrix2, theta=theta)')
        else:
            cov += eval('ak.' + str(ktype) +
                        '_kernel(matrix1, matrix2, theta=theta)')
    if regularization is not None:
        cov += regularization * np.identity(len(cov))
    return cov
