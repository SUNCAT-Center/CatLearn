# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:30:20 2016

@author: mhangaard
"""
from __future__ import absolute_import
from __future__ import division

import numpy as np
from scipy.linalg import cholesky, cho_solve
from .covariance import get_covariance
from .kernels import list2kdict


def log_marginal_likelihood(theta, train_fp, y, kernel_dict):
    """ Return the log marginal likelyhood.
        (Equation 5.8 in C. E. Rasmussen and C. K. I. Williams, 2006)

        Parameters
        ----------
        theta : list
            A list containing the hyperparameters.
        train_fp : list
            A list of the test fingerprint vectors.
        y : list
            A list of target values
        kernel_dict: dict
            A dictionary of kernel dictionaries
    """
    # Get the covariance matrix.
    kernel_dict = list2kdict(theta, kernel_dict)
    K = get_covariance(kernel_dict=kernel_dict, train_matrix=train_fp,
                       regularization=theta[-1])
    n = len(y)
    y = y.reshape([n, 1])
    # print(np.shape(K), np.max(K), np.min(K))
    L = cholesky(K, lower=True)
    a = cho_solve((L, True), y)
    datafit = -.5*np.dot(y.T, a)
    complexity = -np.log(np.diag(L)).sum()  # (A.18) in R. & W.
    normalization = -.5*n*np.log(2*np.pi)
    p = (datafit + complexity + normalization).sum()
    return -p
