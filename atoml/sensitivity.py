"""Function performing GP sensitivity analysis."""
from __future__ import absolute_import
from __future__ import division

import numpy as np

from .covariance import get_covariance


def mean_sensitivity(train_matrix, train_targets, test_matrix, kernel_dict,
                     regularization):
    """Feature sensitivity on the predicted mean.

    Parameters
    ----------
    train_matrix : array
        Training feature space data.
    train_targets : list
        A list of target values.
    test_matrix : array
        Testing feature space data.
    kernel_dict : dict
        Information for the kernel.
    regularization : float
        Noise parameter.
    """
    d, f = np.shape(train_matrix)
    w = kernel_dict['k1']['width']

    # Calculate covariance.
    cvm = get_covariance(kernel_dict=kernel_dict, matrix1=train_matrix,
                         regularization=regularization)
    ktb = get_covariance(kernel_dict=kernel_dict, matrix1=train_matrix,
                         regularization=regularization)

    # Calculate weight estimates.
    cinv = np.linalg.inv(cvm)
    alpha = np.dot(cinv, train_targets)

    # Calculate sensitivities for all features.
    sen = []
    for j in range(f):
        d2 = 0.
        for q in test_matrix:
            d1 = 0.
            for p, ap, kt in zip(train_matrix, alpha, ktb):
                d1 += np.sum(np.dot((np.dot(ap, np.subtract(p[j], q[j])) /
                                     w[j]), kt))
            d2 += 1 / d * (d1 ** 2)
        sen.append(d2)

    return sen
