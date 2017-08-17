"""Function performing uncertainty analysis."""
from __future__ import absolute_import
from __future__ import division

import numpy as np
from scipy.special import erf

from .covariance import get_covariance


def get_uncertainty(kernel_dict, test_fp, reg, ktb, cinv):
    """Function to calculate uncertainty.

    Parameters
    ----------
    kernel_dict : dict
        Dictionary containing all information for the kernels.
    test_fp : array
        Test feature set.
    reg : float
        Regularization parameter.
    ktb : array
        Covariance matrix for test and training data.
    cinv : array
        Covariance matrix for training dataset.
    """
    kxx = get_covariance(kernel_dict=kernel_dict,
                         matrix1=test_fp)
    u = [(reg + kxx[kt][kt] - np.dot(np.dot(ktb[kt], cinv),
                                     np.transpose(ktb[kt]))) **
         0.5 for kt in range(len(ktb))]

    return u


def cdf_fit(x, m, v):
    """Calculate the cumulative distribution function.

    Parameters
    ----------
    x : float
        Known value.
    m : float
        Predicted mean.
    v : float
        Variance on prediction.
    """
    cdf = 0.5 * (1 + erf((x - m) / np.sqrt(2 * v ** 2)))

    return cdf
